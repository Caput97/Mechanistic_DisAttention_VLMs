import torch
import numpy as np

from attn_knockout.utils import (
    build_message_VT,
    inference,
    get_key_positions,
    get_target_sentence_pos,
    get_token_region,
    build_key_spans_for_mode,
    get_query_rows,
    build_mask_ranges,
)

from attn_knockout.patching import (
    set_mask_ranges,
    clear_mask_ranges,
)




# -----------------------------------------------------------------------------
# Tracking attn scores
# -----------------------------------------------------------------------------

def track_attention_layerwise(
    prompt: str,
    video_path: str,
    caption_sent: str,
    foil_sent: str,
    model,
    processor,
    tokenizer,
    mask_mode: str = "none",
    query_scope: str = "assistant_token",
    top_k: int = 3,
):
    """
    Track attention distributions for caption and foil sentences layer-wise
    under a given masking configuration (mask_mode).

    For each label in {caption, foil}, this function:
      - builds the chat message (video + prompt + target sentence),
      - runs preprocessing (processor + vision),
      - computes key token positions (including the target sentence span),
      - builds mask_ranges for the chosen mask_mode and sets them on the model,
      - runs a forward pass with output_attentions=True,
      - aggregates the attention weights for the target sentence tokens:

        For each layer:
          * take attention weights with queries in [text_start:text_end],
          * average over all heads  -> [target_len, seq_len],
          * DO NOT average over target tokens, so we keep one attention
            distribution per target token.

    For each layer and mask_mode we return, per label:
      - attn_tensor: torch.Tensor [target_len x seq_len] (float32, CPU)
      - target_tokens: list of strings (tokens of the assistant sentence)
      - topk: for each target token, top-k most attended input tokens
              (index, token string, score, and region label)
      - region_stats: attention mass for each region (global and per token)

      IMPORTANT FIX:
      - We analyze exactly the same query rows that we masked (qrows=query_rows),
        instead of slicing [text_start:text_end].

        Output alignment:
      - target_tokens: tokens in [text_start:text_end)
      - topk / region_stats: one entry per target token (same indexing)
    """
    results_attn_layerwise = {}

    for label, target_sent in [("caption", caption_sent), ("foil", foil_sent)]:
        # 1) Build the multimodal chat message
        message = build_message_VT(prompt, video_path, target_sent)
        text, inputs, video_inputs, video_kwargs = inference(message, processor, model)

        # 2) Token positions
        positions = get_key_positions(inputs, tokenizer)

        assistant_idx = positions.get("assistant")
        assistant_nl_idx = positions.get("assistant\n")
        if assistant_nl_idx is None:
            if assistant_idx is None:
                raise RuntimeError("Could not locate assistant token indices in input.")
            assistant_nl_idx = assistant_idx + 1

        # 3) Span of the assistant target sentence tokens
        text_start, text_end = get_target_sentence_pos(
            tokenizer, processor, inputs, assistant_nl_idx
        )
        if text_start is None or text_end is None or text_start >= text_end:
            print("[WARN] Empty target span, skipping attention tracking for this label.")
            continue

        # Full input tokens
        input_ids = inputs["input_ids"][0]
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        seq_len = input_ids.size(0)

        # Tokens you want to OUTPUT (actual sentence tokens)
        target_tokens = input_tokens[text_start:text_end]
        target_len = len(target_tokens)

        # 4) Build masking ranges and set them
        key_spans = build_key_spans_for_mode(positions, mask_mode)
        query_rows = get_query_rows(query_scope, positions, text_start, text_end)
        mask_ranges = build_mask_ranges(query_rows, key_spans)
        set_mask_ranges(model, mask_ranges)

        # We ANALYZE attention on the SAME predictive rows we masked.
        # For correct 1:1 mapping to each target token:
        #   predictive row for token at position p is (p-1).
        # So we use qrows = [text_start-1, text_start, ..., text_end-2]
        # which has length == target_len.
        # For assistant_token: only the first token is meaningful.
        if query_scope == "assistant_token":
            qrows = [text_start - 1]
        else:
            qrows = list(range(text_start - 1, text_end - 1))  # ends at text_end-2
        # For logging / debugging
        query_rows_used = qrows.copy()

        # Sanity: align sizes
        # assistant_content: len(qrows) == target_len
        # assistant_token:   len(qrows) == 1 (we will only compute 1 token stats)
        num_layers = len(model.model.layers)
        label_results = {}

        # Column regions
        region_labels = [get_token_region(j, positions) for j in range(seq_len)]
        regions = ["video", "user_text", "A", "B", "other"]
        region_indices = {
            r: [j for j, lab in enumerate(region_labels) if lab == r]
            for r in regions
        }

        with torch.no_grad():
            output = model(
                **inputs,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=False,
            )

            attentions = output.attentions  # list[tensor], one per layer

            for layer_idx, attn in enumerate(attentions):
                attn_layer = attn[0]  # [num_heads, seq_len, seq_len]

                # Query rows: predictive rows (masked rows)
                query_attn = attn_layer[:, qrows, :]  # [heads, Q, seq_len]
                query_attn = query_attn.to(torch.float32)
                mean_over_heads = query_attn.mean(dim=0)  # [Q, seq_len]
                mean_np = mean_over_heads.detach().cpu().numpy()

                # Build per-target-token outputs
                # If assistant_content: Q == target_len -> perfect 1:1
                # If assistant_token: Q == 1 -> we only output stats for token 0, others left empty/None
                topk_info = {}
                region_stats = {"global": {}, "per_token": {}}

                # Global: over available query rows
                total_mass = float(mean_np.sum()) + 1e-8
                for r in regions:
                    idxs = region_indices[r]
                    if not idxs:
                        region_stats["global"][r] = 0.0
                    else:
                        mass = float(mean_np[:, idxs].sum())
                        region_stats["global"][r] = mass / total_mass

                # Per-token:
                if query_scope == "assistant_content":
                    # One row per target token
                    for t_idx in range(target_len):
                        row = mean_np[t_idx]
                        top_k_eff = min(top_k, seq_len)
                        top_indices = row.argsort()[::-1][:top_k_eff]

                        token_topk = []
                        for j in top_indices:
                            token_topk.append(
                                {
                                    "idx": int(j),
                                    "token": input_tokens[j],
                                    "score": float(row[j]),
                                    "region": region_labels[j],
                                }
                            )
                        topk_info[str(t_idx)] = token_topk

                        row_sum = float(row.sum()) + 1e-8
                        per_tok = {}
                        for r in regions:
                            idxs = region_indices[r]
                            if not idxs:
                                per_tok[r] = 0.0
                            else:
                                per_tok[r] = float(row[idxs].sum()) / row_sum
                        region_stats["per_token"][str(t_idx)] = per_tok

                    # attn_tensor for saving: [target_len, seq_len]
                    attn_tensor = mean_over_heads.detach().cpu()

                else:
                    # assistant_token: we only have 1 row (for the first target token)
                    row = mean_np[0]
                    top_k_eff = min(top_k, seq_len)
                    top_indices = row.argsort()[::-1][:top_k_eff]
                    token_topk = []
                    for j in top_indices:
                        token_topk.append(
                            {
                                "idx": int(j),
                                "token": input_tokens[j],
                                "score": float(row[j]),
                                "region": region_labels[j],
                            }
                        )
                    topk_info["0"] = token_topk

                    row_sum = float(row.sum()) + 1e-8
                    per_tok = {}
                    for r in regions:
                        idxs = region_indices[r]
                        if not idxs:
                            per_tok[r] = 0.0
                        else:
                            per_tok[r] = float(row[idxs].sum()) / row_sum
                    region_stats["per_token"]["0"] = per_tok

                    # attn_tensor for saving: [1, seq_len]
                    attn_tensor = mean_over_heads.detach().cpu()

                layer_key = f"layer_{layer_idx}"
                label_results[layer_key] = {
                    "attn_tensor": attn_tensor,      # [Q x seq_len]
                    "target_tokens": target_tokens,  # only actual sentence tokens
                    "query_rows_used": query_rows_used,
                    "topk": topk_info,               # aligned to target token indices
                    "region_stats": region_stats
                }

        results_attn_layerwise[label] = label_results
        clear_mask_ranges(model)

    return results_attn_layerwise