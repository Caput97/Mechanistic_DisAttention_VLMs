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
