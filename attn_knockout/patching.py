import torch
from types import MethodType
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention


#PER ORA LASCIO COSì, QUANDO AGGIUNGERO' ALTRI MODELLI:
#def patch_model(model, model_name):
#    if "qwen" in model_name:
#        patch_qwen(model)
#    elif "llava" in model_name:
#        patch_llava(model)

# -----------------------------------------------------------------------------
# Attention patching (knockout of specific attention flows)
# -----------------------------------------------------------------------------

# Keep a reference to the original forward method
_orig_forward = Qwen2_5_VLAttention.forward


def patched_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """
Custom forward for Qwen2_5_VLAttention.

This patch:
  1. Ensures that a full (batch, 1, seq, seq) attention_mask exists.
     - If None, a standard lower-triangular causal mask is created from scratch.
     - Otherwise, the provided mask is cloned (and cast to the correct dtype)
       to avoid in-place modifications of shared tensors.
  2. Applies custom attention blocking using self.mask_ranges, a list of
     (row_idx, start_idx, end_idx) tuples:
       - For each tuple, the row corresponding to row_idx is prevented from
         attending to all token positions in the interval [start_idx, end_idx).
       - Indices are clamped to valid bounds to avoid out-of-range access.
       - Self-attention for row_idx (i.e., attending to itself) is explicitly
         preserved by setting its diagonal value back to 0.0.
  3. If self.mask_ranges is empty or not set, no additional blocking is applied
     and the standard causal attention behavior is preserved.
  4. Calls the *original* attention forward with the modified attention_mask
     and returns its outputs unchanged.

Requirements:
  - self.mask_ranges must be set externally before the forward call if
    custom masking is desired.
  - Each element of self.mask_ranges must be a tuple:
        (row_idx, start_idx, end_idx),
    indicating which range of keys should be masked for a specific query row.
"""

    bsz, seq_len, _ = hidden_states.size()

    # Read mask ranges from the attention module (set externally)
    mask_ranges = getattr(self, "mask_ranges", None)

    # If attention_mask is None, build a standard causal mask (lower triangular)
    if attention_mask is None:

        # create lower‑triangular full zeros/ones
        causal = torch.tril(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
        )

        # transform in bias pre‑softmax: 0 to allow attention, -inf to stop it
        causal = causal.masked_fill(causal == 0, float("-inf")).masked_fill(causal == 1, 0.0)

        # shape  as (batch,1,seq,seq)
        attention_mask = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
    else:
        # Clone to avoid in-place modifications on shared tensors
        attention_mask = attention_mask.clone().to(hidden_states.dtype)


    # Inject custom blocking ranges, if any
    if mask_ranges:
        for (row_idx, start_idx, end_idx) in mask_ranges:
            if row_idx is None:
                continue
            if row_idx < 0 or row_idx >= seq_len:
                continue

            # Clamp indices to valid range
            start = max(0, min(start_idx, seq_len))
            end   = max(0, min(end_idx,   seq_len))
            if start < end:
                # Block attention from row_idx to tokens in [start, end)
                attention_mask[:, :, row_idx, start:end] = float("-inf")
                # Keep self-attention on row_idx
                attention_mask[:, :, row_idx, row_idx] = 0.0

    # Call the original attention forward and return all outputs
    attn_output, attn_weights, past = _orig_forward(
        self,
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs
    )

    return attn_output, attn_weights, past





def set_mask_ranges(model, mask_ranges):
    """
    Set the same list of (row_idx, start_idx, end_idx) on all
    Qwen2_5_VLAttention modules in the model.
    These attributes are then used inside patched_forward() to modify the attention mask.
    """
    for m in model.modules():
        if isinstance(m, Qwen2_5_VLAttention):
            m.mask_ranges = mask_ranges


#for the no-blocking case
def clear_mask_ranges(model):
    """
    Clear any custom blocking, returning to the standard causal attention.
    """
    set_mask_ranges(model, [])


def patch_all_attention_layers(model):
    """
    Monkey-patch the forward method of all Qwen2_5_VLAttention modules (sel-attention layers)
    and initialize the "mask_ranges" attribute to an empty list, while storing
    the original forward in 'attn._original_forward.
    """
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn._original_forward = attn.forward
        attn.forward = MethodType(patched_forward, attn)
        attn.layer_index = i
        attn.mask_ranges = []  # by default: no extra blocking
        print(f"Layer {i} patched:", attn.forward.__func__ is patched_forward)
    print("✅ All attention layers have been patched.\n")