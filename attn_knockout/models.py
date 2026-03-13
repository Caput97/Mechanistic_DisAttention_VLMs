'''
from typing import Tuple

import torch
from transformers import AutoProcessor

# Qwen2.5-VL
from transformers import Qwen2_5_VLForConditionalGeneration

# LLaVA-NeXT loader (solo quando serve)
# (non importarlo globalmente se vuoi evitare import error quando llava non è installato)


def load_model(
    model_name: str,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
):
    """
    Factory for loading multimodal models.

    Returns:
        model
        processor
        tokenizer
        lm_head
    """
    model_name_l = model_name.lower()

    if "qwen2.5" in model_name_l or "qwen2_5" in model_name_l or "qwen2-5" in model_name_l:
        return _load_qwen(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

    if "llava" in model_name_l:
        return _load_llava_onevision(
            model_name,  # puoi passare l'hf id direttamente
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

    raise ValueError(f"Unsupported model: {model_name}")


# -------------------------
# Model-specific loaders
# -------------------------

def _load_qwen(
    model_id: str,
    torch_dtype,
    device_map,
    attn_implementation,
):
    print(f"🔹 Loading Qwen model: {model_id}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    tokenizer = processor.tokenizer
    lm_head = model.lm_head

    print("✅ Qwen loaded.")
    return model, processor, tokenizer, lm_head


def _load_llava_onevision(
    model_id: str,
    device_map: str,
    attn_implementation,
):
    """
    LLaVA-OneVision (LLaVA-NeXT) loader.
    Nota: torch_dtype/attn_implementation non sono gestiti allo stesso modo del loader Qwen2.5-VL.
    """
    print(f"🔹 Loading LLaVA-OneVision model: {model_id}")

    # Import lazy per non rompere l'env se llava non è installato
    from llava.model.builder import load_pretrained_model

    # Per questo checkpoint specifico, da doc:
    # model_name = "llava_qwen"
    # (il warning 'llava vs llava_qwen' lo hai già visto: è ok in pratica per questo model)
    model_name = "llava_qwen"

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_id,
        None,
        model_name,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    model.eval()

    # Nel tuo codice Qwen "processor" è un AutoProcessor con .tokenizer.
    # Qui non esiste AutoProcessor: usiamo image_processor come "processor"
    # e manteniamo tokenizer separato (come già fai nel return).
    processor = image_processor

    # lm_head: per LlavaQwenForCausalLM il decoder è Qwen2-like -> dovrebbe esistere .lm_head
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise AttributeError("LLaVA model has no lm_head attribute (unexpected for LlavaQwenForCausalLM).")

    print("✅ LLaVA-OneVision loaded.")
    return model, processor, tokenizer, lm_head


'''



from typing import Tuple

import torch
from transformers import AutoProcessor

# Qwen
from transformers import Qwen2_5_VLForConditionalGeneration


def load_model(
    model_name: str,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
):
    """
    Factory for loading multimodal models.

    Returns:
        model
        processor
        tokenizer
        lm_head
    """

    model_name = model_name.lower()

    if "qwen" in model_name:
        return _load_qwen(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

    # elif "llava" in model_name:
    #     return _load_llava(...)

    else:
        raise ValueError(f"Unsupported model: {model_name}")


# -------------------------
# Model-specific loaders
# -------------------------





def _load_qwen(
    model_id: str,
    torch_dtype,
    device_map,
    attn_implementation,
):
    print(f"🔹 Loading Qwen model: {model_id}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    tokenizer = processor.tokenizer
    lm_head = model.lm_head

    print("✅ Qwen loaded.")
    return model, processor, tokenizer, lm_head    


