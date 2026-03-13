import math
import torch
import torch.nn.functional as F

from attn_knockout.utils import (
    build_message_VT,
    inference,
    get_key_positions,
    get_target_sentence_pos,
    build_key_spans_for_mode,
    get_query_rows,
    build_mask_ranges,
)

from attn_knockout.patching import (
    set_mask_ranges,
    clear_mask_ranges,
)








# -----------------------------------------------------------------------------
# Scoring utilities
# -----------------------------------------------------------------------------


def get_target_sent_score_from_logits(
    logits: torch.Tensor,
    inputs,
    text_start: int,
    text_end: int,
    mode: str = "mean",
):
    """
    Compute a score for the target sentence given model logits.

    Args:
        logits: tensor of shape [seq_len, vocab_size]
        inputs: original model inputs (to recover token ids)
        text_start: starting index of target sentence
        text_end: ending index (exclusive) of target sentence
        mode:
          - "sum":  sum of log-probabilities over sentence tokens
          - "mean": mean log-prob (length-normalized)
          - "ppl":  perplexity (exp(-mean log-prob), lower is better)
          - "prob": absolute sentence probability (exp(total log-prob)

    Returns:
        A scalar float score.
    """
    ids = inputs.input_ids[0]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    target_ids = ids[text_start:text_end]
    positions = torch.arange(text_start, text_end, device=ids.device)

    # For a causal LM, logits[i-1] predicts token i
    token_lp = log_probs[positions - 1, target_ids]


    if mode == "sum":
        total_logprob = token_lp.sum().item()
        return total_logprob

    elif mode == "mean":
        mean_logprob  = token_lp.mean().item()
        return mean_logprob

    elif mode == "ppl":
        mean_logprob  = token_lp.mean().item()
        return float(math.exp(-mean_logprob))

    elif mode == "prob":
        total_logprob = token_lp.sum().item()
        # absolute sentence probability
        prob = math.exp(total_logprob)

        # protect vs floating underflow / NaN
        if math.isnan(prob) or prob == 0.0:
            return 0.0

        return prob

    else:
        raise ValueError(
            "mode must be one of {'sum', 'mean', 'ppl', 'prob'}"
        )





# -----------------------------------------------------------------------------
# Tracking functions (sentence probabilities across layers / positions)
# -----------------------------------------------------------------------------



def track_sentence_prob_layerwise(
    prompt: str,
    video_path: str,
    caption_sent: str,
    foil_sent: str,
    model,
    processor,
    tokenizer,
    lm_head,
    mask_mode: str = "none",
    query_scope: str = "assistant_token",
):
    """
    Track the score of caption and foil sentences layer-wise, under a given
    masking configuration (mask_mode).

    mask_mode : {"none", "user", "vision", "user_text", "A_content", "B_content"}

    For each label in {caption, foil}, we:
      - build the chat message,
      - run preprocessing (processor + vision),
      - compute token positions,
      - build mask_ranges for the chosen mode and set them on the model,
      - For each transformer layer, take the hidden states, apply the final norm,
        then the lm_head, and compute a sentence score from the resulting logits
        using the default mode defined in get_target_sent_score_from_logits().

    For a single masking configuration (mask_mode), return:

    {
      "caption": { "layer_0": [score], "layer_1": [score], ... },
      "foil":    { "layer_0": [score], "layer_1": [score], ... }
    }

    So the structure is still layer-wise and per label; the inner key is the
    mask_mode (e.g., "none", "vision", etc.).
    """
    results_layerwise = {}

    for label, target_sent in [("caption", caption_sent), ("foil", foil_sent)]:
        message = build_message_VT(prompt, video_path, target_sent)
        text, inputs, video_inputs, video_kwargs = inference(message, processor, model)

        # 1) Get all key positions from the new get_key_positions()
        positions = get_key_positions(inputs, tokenizer)


        # 2) Compute the target sentence span using 'assistant\\n'
        #assistant_nl_idx = positions["assistant\n"]

        assistant_idx = positions.get("assistant")
        assistant_nl_idx = positions.get("assistant\n")

        if assistant_nl_idx is None:
            if assistant_idx is None:
                raise RuntimeError("Could not locate assistant token indices in input.")
            assistant_nl_idx = assistant_idx + 1  

        
        start, end = get_target_sentence_pos(
            tokenizer,
            processor,
            inputs,
            assistant_nl_idx,
        )

        

        # 3) Build attention mask ranges for the chosen masking configuration

        #Build KEY spans (columns) for this mask_mode
        key_spans = build_key_spans_for_mode(positions, mask_mode)
        #Build QUERY rows (which tokens' attention is affected) based on query_scope
        query_rows = get_query_rows(query_scope, positions, start, end)

        #Combine them into (row, key_start, key_end) tuples (== cross-product)
        mask_ranges = build_mask_ranges(query_rows, key_spans)
        #Set on model (read by patched_forward)
        set_mask_ranges(model, mask_ranges)



        # One dict per transformer layer
        num_layers = len(model.model.layers)
        layer_probs = {f"layer_{i}": [] for i in range(num_layers)}

        with torch.no_grad():
            # Single forward pass (mask_mode is global for this call)

            output = model(
                **inputs,
                use_cache=False,
                #True only if you want to inspect attentions
                #output_attentions=True,
                output_attentions=False,
                output_hidden_states=True,
            )

            hidden_states = output.hidden_states  # hidden_states[0]=embedding, [1]=layer0, ...
            norm = model.model.norm

            # We'll store scores under the key "mask_mode" to remember which setting was used
            key_name = mask_mode

            # For each layer, compute the sentence score
            # based on the normalized (or last) hidden state
            for layer_idx in range(1, len(hidden_states)):
                h = hidden_states[layer_idx][0]

                # Apply final norm to all layers except the last one
                if layer_idx != len(hidden_states) - 1:
                    h_norm = norm(h)
                else:
                    h_norm = h

                logits = lm_head(h_norm)
                score = get_target_sent_score_from_logits(
                    logits,
                    inputs,
                    start,
                    end
                )




                layer_key = f"layer_{layer_idx - 1}"
                layer_probs[layer_key].append(score)

        results_layerwise[label] = layer_probs
    
    # Reset any custom blocking to get back to the baseline causal mask
    clear_mask_ranges(model)

    return results_layerwise

