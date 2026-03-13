import re
import torch
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info


#quando torno a modalità video, devo cambiare tutti i image_pad in video_pad dove sono tra virgolette. il resto dei nomi variabili l'ho mantenuto a video_pad 
# -----------------------------------------------------------------------------
# Message builders (video + text chat format)
# -----------------------------------------------------------------------------

#version of prompt with image + text

def build_message_VT(prompt_text: str, video_path: str, target_sent: str):
    """
    Build a chat message where the user provides:
      - an image
      - a multiple-choice prompt
    and the assistant provides the target answer (caption or foil).

    This format is used to compute probabilities for the target sentence
    given video + prompt as context.
    """
    message = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": video_path,
                    "max_pixels": 360 * 420,
                    
                },
                {"type": "text", "text": prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": target_sent},
            ],
        },
    ]
    return message

'''
#final version for VT generation (prompt + video as input, generation of A/B as output)

def build_message_VT(prompt_text: str, video_path: str, target_sent: str):
    """
    Build a chat message where the user provides:
      - a video
      - a multiple-choice prompt
    and the assistant provides the target answer (caption or foil).

    This format is used to compute probabilities for the target sentence
    given video + prompt as context.
    """
    message = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    #32 frames
                    #"fps": 1.1, 
                    #16 frames
                    "fps": 0.55,
                    #20 frames
                    #"fps": 0.69,
                },
                {"type": "text", "text": prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": target_sent},
            ],
        },
    ]
    return message

'''


'''
#build the message text+video

def build_message_TV(prompt_text, video_path):

    message = [

    {"role": "system", "content": [ {"type": "text", "text": "You are a helpful assistant."},] },

    {"role": "user", "content": [ 
        {"type": "text", "text": prompt_text},

        {"type": "video", 
         "video": video_path, 
         "max_pixels": 360 * 420,
         #"fps": 1.1,
        },
                
        {"type": "text", "text": "Qual è la risposta corretta?"}],
    }

    ]
    return message
'''




# -----------------------------------------------------------------------------
# Pre-processing / inference utilities
# -----------------------------------------------------------------------------

'''
#official inference function (with chat template + vision processing)

def inference(message, processor, model):
    """
    Apply the chat template and process both text and video inputs.

    Returns:
        text: formatted text string from the chat template
        inputs: processed inputs ready for the model
        video_inputs: processed video tensors
        video_kwargs: additional video-related kwargs
    """
    # Apply chat template (no tokenization yet, just formatting)
    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=False
    )
    #print("Formatted text:", text)

    # Extract image/video info for the processor
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        message,
        return_video_kwargs=True
    )

    #print(type(video_inputs), type(image_inputs))

    #debug_vision_inputs(image_inputs, video_inputs)

    # Build full input batch for the model (single example)
    # --> Call processor with formatted text and extracted vision data
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    return text, inputs, video_inputs, video_kwargs
'''

#inference with images
def inference(message, processor, model):
    """
    Apply the chat template and process both text and image inputs.

    Returns:
        text: formatted text string from the chat template
        inputs: processed inputs ready for the model
        video_inputs: processed video tensors
        video_kwargs: additional video-related kwargs
    """
    # Apply chat template (no tokenization yet, just formatting)
    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=False
    )
    #print("Formatted text:", text)

    # Extract image/video info for the processor
    image_inputs, video_inputs = process_vision_info(
        message
    )

    #print(type(video_inputs), type(image_inputs))

    # --> Call processor with formatted text and extracted vision data
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return text, inputs, video_inputs, None



def get_video_id(video_id_str: str) -> str:
    """
    Extract the first numerical sequence from a video_id string.

    Example:
        "video_0123_temp" -> "0123"
    """
    match = re.search(r"\d+", video_id_str)
    if match:
        return match.group()
    raise ValueError(f"No digits found in video_id='{video_id_str}'")






# -----------------------------------------------------------------------------
# Sequence / token position helpers
# -----------------------------------------------------------------------------

def get_target_sentence_pos(tokenizer, processor, inputs, assistant_role_idx: int):
    """
    Given the token index where the assistant role token ends (i.e. 'assistant\\n'),
    find the start and end positions (token indices) of the target sentence.

    The target sentence is assumed to be:
      <|im_start|> assistant \\n [TARGET SENTENCE TOKENS] <|im_end|>
    """
    ids = inputs.input_ids[0]
    tokens = processor.tokenizer.convert_ids_to_tokens(ids.tolist())

    # Target text starts right after "assistant\\n"
    text_start = assistant_role_idx + 1

    # Find the next <|im_end|> that closes the assistant message
    im_end_sentence = next(
        (j for j in range(text_start, len(tokens)) if tokens[j] == "<|im_end|>"),
        None,
    )

    text_end = im_end_sentence
    return text_start, text_end


def get_key_positions(inputs, tokenizer):
    """
    Identify key token positions in the input sequence, including:

    - "vision_start": index of "<|vision_start|>"
    - "vision_end": index of "<|vision_end|>"
    - "user_im_end": index of "<|im_end|>" closing the user message
    - "assistant": index of the 'assistant' token in the assistant message
    - "assistant\n": index of the token right after 'assistant' (usually '\n')
    - "A": index of the 'A' token in option "A."
    - "B": index of the 'B' token in option "B."
    - "A_content_start"/"A_content_end": [start, end) range for the content of A
    - "B_content_start"/"B_content_end": [start, end) range for the content of B
    - "video_pad": list of indices of "<|video_pad|>" tokens
    """

    input_ids = inputs["input_ids"]
    assert input_ids.size(0) == 1, "Batch size > 1 is not supported here."

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)
    positions = {}

    def clean(tok: str) -> str:
        """Remove BPE/whitespace prefixes like Ġ or ▁."""
        return tok.lstrip("Ġ▁")

    # Vision start / end
    positions["vision_start"] = next(
        (i for i, t in enumerate(tokens) if t == "<|vision_start|>"), None
    )
    positions["vision_end"] = next(
        (i for i, t in enumerate(tokens) if t == "<|vision_end|>"), None
    )

    # All video_pad positions
    positions["image_pad"] = [i for i, t in enumerate(tokens) if t == "<|image_pad|>"]

    # All <|im_start|> and <|im_end|> positions
    im_start_positions = [i for i, t in enumerate(tokens) if t == "<|im_start|>"]
    im_end_positions   = [i for i, t in enumerate(tokens) if t == "<|im_end|>"]

    # The last <|im_start|> corresponds to the assistant message
    last_im_start = im_start_positions[-1] if im_start_positions else None

    # The <|im_end|> of the user is the last <|im_end|> before that
    user_im_end = None
    if last_im_start is not None:
        prev_ends = [i for i in im_end_positions if i < last_im_start]
        user_im_end = prev_ends[-1] if prev_ends else None
    positions["user_im_end"] = user_im_end

    # 'assistant' token and the following newline
    assistant_idx = None
    if last_im_start is not None:
        for i in range(last_im_start, seq_len):
            if clean(tokens[i]) == "assistant":
                assistant_idx = i
                break
    positions["assistant"] = assistant_idx
    positions["assistant\n"] = (
        assistant_idx + 1
        if assistant_idx is not None and assistant_idx + 1 < seq_len
        else None
    )

    # Option A/B ("A." / "B.")
    A_idx = None
    B_idx = None
    for i in range(seq_len - 1):
        if clean(tokens[i]) == "A" and clean(tokens[i + 1]) == ".":
            A_idx = i
        if clean(tokens[i]) == "B" and clean(tokens[i + 1]) == ".":
            B_idx = i

    positions["A"] = A_idx
    positions["B"] = B_idx

    # Content of A: after "A" "." up to "B"
    if A_idx is not None and B_idx is not None:
        positions["A_content_start"] = A_idx + 2     # skip 'A' and '.'
        positions["A_content_end"]   = B_idx         # exclusive (before 'B')
    else:
        positions["A_content_start"] = None
        positions["A_content_end"]   = None

    # Content of B: after "B" "." up to <|im_end|> of the user
    if B_idx is not None and user_im_end is not None:
        positions["B_content_start"] = B_idx + 2
        positions["B_content_end"]   = user_im_end   # exclusive (before <|im_end|>)
    else:
        positions["B_content_start"] = None
        positions["B_content_end"]   = None

    return positions

#compute the key spans to be masked for a given mode
def build_key_spans_for_mode(positions, mode: str):
    """
    Returns a list of (key_start, key_end) spans to be masked (end exclusive).
    """
    vs = positions.get("vision_start", None)
    ve = positions.get("vision_end", None)
    user_im_end = positions.get("user_im_end", None)

    A_start = positions.get("A_content_start", None)
    A_end   = positions.get("A_content_end", None)
    B_start = positions.get("B_content_start", None)
    B_end   = positions.get("B_content_end", None)

    video_pads = positions.get("image_pad", [])

    if mode == "none":
        return []

    if mode == "user":
        if vs is None or user_im_end is None:
            return []
        return [(vs, user_im_end)]

    if mode == "vision":
        if vs is None or ve is None:
            return []
        return [(vs, ve + 1)]

    if mode == "user_text":
        if ve is None or user_im_end is None:
            return []
        text_start = ve + 1
        if text_start >= user_im_end:
            return []
        return [(text_start, user_im_end)]

    if mode == "A_content":
        if A_start is None or A_end is None or A_start >= A_end:
            return []
        return [(A_start, A_end)]

    if mode == "B_content":
        if B_start is None or B_end is None or B_start >= B_end:
            return []
        return [(B_start, B_end)]

    if mode == "vision_half1":
        if not video_pads:
            return []
        n = len(video_pads)
        half = n // 2
        if half == 0:
            return []
        first_pad = video_pads[0]
        last_blocked_pad = video_pads[half - 1]
        return [(first_pad, last_blocked_pad + 1)]

    if mode == "vision_half2":
        if not video_pads:
            return []
        n = len(video_pads)
        half = n // 2
        if half == 0:
            return []
        first_blocked_pad = video_pads[half]
        last_pad = video_pads[-1]
        return [(first_blocked_pad, last_pad + 1)]

    raise ValueError(f"Unknown mode '{mode}'")

#compute the query rows to be masked for a given scope
def get_query_rows(query_scope: str, positions, text_start: int, text_end: int):
    """
    query_scope:
      - "assistant_token": mask only one row (row that predicts the 1st target token)
      - "assistant_content": mask all rows that predict tokens in [text_start, text_end)
    """
    if query_scope == "assistant_token":
        row = text_start - 1
        return [row] if row is not None else []

    if query_scope == "assistant_content":
        row_start = text_start - 1
        row_end_excl = text_end - 1
        if row_start is None or row_end_excl is None or row_start >= row_end_excl:
            return []
        return list(range(row_start, row_end_excl))

    raise ValueError(f"Unknown query_scope '{query_scope}'")

#compute the final mask_ranges for a given mode by combining query rows and key spans
def build_mask_ranges(query_rows, key_spans):
    """
    Cartesian product between query_rows and key_spans -> list of (row_idx, start_idx, end_idx)
    """
    if not query_rows or not key_spans:
        return []
    out = []
    for r in query_rows:
        for (ks, ke) in key_spans:
            out.append((r, ks, ke))
    return out

# --> recap:
#build_key_spans_for_mode
# == what we are masking (vision, user_text, A, B, …)
#get_query_rows_for_scope
# == where we are masking it (only 1 row, or all assistant content rows)
#build_mask_ranges
# == cross-product of the above two


def get_token_region(idx: int, positions: dict) -> str:
    """
    Return a coarse region label for a given token index, based on the
    positions dictionary computed by `get_key_positions`.

    Possible regions:
      - "video": token belongs to <|video_pad|> segment
      - "A": token inside option A content span
      - "B": token inside option B content span
      - "user_text": other tokens in the user message (between vision_end and user_im_end)
      - "other": system tokens, assistant tokens, or anything not covered above
    """
    video_pads = set(positions.get("image_pad", []))
    A_start = positions.get("A_content_start", None)
    A_end   = positions.get("A_content_end", None)
    B_start = positions.get("B_content_start", None)
    B_end   = positions.get("B_content_end", None)
    vision_end = positions.get("vision_end", None)
    user_im_end = positions.get("user_im_end", None)

    if idx in video_pads:
        return "video"

    if A_start is not None and A_end is not None and A_start <= idx < A_end:
        return "A"

    if B_start is not None and B_end is not None and B_start <= idx < B_end:
        return "B"

    if vision_end is not None and user_im_end is not None and vision_end < idx < user_im_end:
        return "user_text"

    return "other"


#########################################
#Generation utilities
#########################################

# --- NEW: A/B prompt + inference + layerwise A/B tracking ---

import math
import torch
import torch.nn.functional as F

from attn_knockout.patching import set_mask_ranges, clear_mask_ranges


def build_ab_prompt(prompt_text: str) -> str:
    """
    Standardizziamo SEMPRE così:
      - istruzione: 'Rispondi solo A o B.'
      - anchor: 'Risposta:'
    """
    base = (prompt_text or "").rstrip()
    return base + "\nRispondi solo A o B.\nRisposta:\n"


def build_message_VT_gen(prompt_text: str, video_path: str):
    """
    Messaggio chat con SYSTEM + USER (video+testo) e poi generation prompt per ASSISTANT.
    Nota: NON mettiamo contenuto assistant, perché vogliamo la distribuzione del *primo token generato*.
    """
    message = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 0.55,
                },
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    return message


def inference_gen(message, processor, model):
    """
    Come inference(), ma con add_generation_prompt=True per ottenere i token dell'assistant "vuoti",
    così il modello predice il prossimo token (A/B) subito dopo.
    """
    text = processor.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,   # <-- QUI la differenza
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    return text, inputs, video_inputs, video_kwargs


def _get_single_token_id(tokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected '{s}' to be single-token, got ids={ids}")
    return ids[0]


def _get_ab_token_ids(tokenizer):
    a_id = _get_single_token_id(tokenizer, "A")
    b_id = _get_single_token_id(tokenizer, "B")
    return a_id, b_id


def _layerwise_ab_from_hidden_states(model, lm_head, hidden_states, tokenizer):
    """
    hidden_states: output.hidden_states (tuple)
      hidden_states[0]=embeddings, hidden_states[1]=layer0, ...
    Noi vogliamo i logits del NEXT token => usiamo la rappresentazione dell'ULTIMO token in input (index -1).
    """
    a_id, b_id = _get_ab_token_ids(tokenizer)

    # nel tuo tracking_prob: norm applicata a tutti tranne l'ultimo hidden_state
    norm = model.model.norm

    num_layers = len(model.model.layers)
    out = {f"layer_{i}": {} for i in range(num_layers)}

    # loop come nel tuo codice: layer_idx in range(1, len(hidden_states)) => layer_0..layer_{n-1}
    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx][0]         # [seq, hidden]
        last = h[-1]                             # [hidden] ultimo token in input

        if layer_idx != len(hidden_states) - 1:
            last = norm(last)
        # else: ultimo layer già "finale"

        logits = lm_head(last)                   # [vocab]
        logit_a = logits[a_id]
        logit_b = logits[b_id]

        # softmax SOLO su {A,B}
        ab = torch.stack([logit_a, logit_b], dim=0)
        p = F.softmax(ab.float(), dim=0)

        layer_key = f"layer_{layer_idx - 1}"
        out[layer_key] = {
            "logit_A": float(logit_a.item()),
            "logit_B": float(logit_b.item()),
            "p_A": float(p[0].item()),
            "p_B": float(p[1].item()),
            "margin": float((logit_a - logit_b).item()),
            "pred": "A" if logit_a.item() >= logit_b.item() else "B",
        }

    return out


def track_ab_prob_layerwise(
    prompt: str,
    video_path: str,
    model,
    processor,
    tokenizer,
    lm_head,
    mask_mode: str = "none",
    query_scope: str = "assistant_token",
):
    """
    Per un dato prompt+video:
      - costruisce chat con generation prompt
      - calcola positions
      - maschera secondo mask_mode & query_scope
      - fa 1 forward con hidden_states
      - ritorna dict layerwise con p_A, p_B, margin, pred
    """
    prompt_ab = build_ab_prompt(prompt)

    # 1) build message + inputs (generation prompt)
    message = build_message_VT_gen(prompt_ab, video_path)
    text, inputs, video_inputs, video_kwargs = inference_gen(message, processor, model)

    # IMPORTANT: manda su device come fai altrove (qui non era nel tuo inference originale)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # 2) positions per span A/B + vision/user ecc.
    positions = get_key_positions(inputs, tokenizer)

    # 3) definiamo "text_start" come la posizione del PRIMO token che verrà generato.
    #    Siccome usiamo add_generation_prompt=True e non abbiamo ancora generato nulla,
    #    il NEXT token sta in posizione seq_len (non esiste ancora nei input_ids).
    seq_len = inputs["input_ids"].shape[1]
    text_start = seq_len
    text_end = seq_len + 1  # serve solo per la logica di get_query_rows

    # 4) build mask ranges (riuso identico delle tue funzioni)
    key_spans = build_key_spans_for_mode(positions, mask_mode)
    query_rows = get_query_rows(query_scope, positions, text_start, text_end)
    mask_ranges = build_mask_ranges(query_rows, key_spans)

    set_mask_ranges(model, mask_ranges)

    # 5) forward
    with torch.no_grad():
        output = model(
            **inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
        )

    hidden_states = output.hidden_states
    layerwise = _layerwise_ab_from_hidden_states(model, lm_head, hidden_states, tokenizer)

    clear_mask_ranges(model)
    return layerwise
