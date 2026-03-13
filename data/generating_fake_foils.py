import os
import re
import json
import time
import argparse
from typing import Any, Dict, List

from tqdm import tqdm
from openai import OpenAI

# =========================
# CHECK API KEY + CLIENT
# =========================
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError(
        "❌ OPENAI_API_KEY non trovata. "
        "Esegui prima: export OPENAI_API_KEY='sk-...'"
    )

client = OpenAI()
print("🔑 OpenAI API key caricata correttamente!")

# =========================
# DEFAULTS
# =========================
DEFAULT_MODEL = "gpt-5-nano"

# =========================
# PROMPTS
# =========================
SYSTEM_PROMPT = (
    "You generate sentences that are plausible in the real world but incompatible with the content "
    "of a video, using only the provided caption as reference."
)

USER_PROMPT_TEMPLATE = """CAPTION:
"{caption}"

Step 1: Identify up to five concrete elements (objects, people, or locations) explicitly mentioned in the caption.

Step 2: Select exactly ONE of those elements.

Step 3: Generate a FOIL sentence that:
- describes a completely different event from the caption,
- reuses ONLY the selected element,
- does not contain any other concrete elements from the caption,
- is plausible in the real world,
- is incompatible with the video content based on the caption provided.

Return exactly one sentence.
Do not include explanations, reasoning steps, lists, quotation marks, or any additional text.
Output only the final FOIL sentence.
"""

# =========================
# OPENAI HELPERS (with retry)
# =========================
def _call_openai(system_prompt: str, user_prompt: str, model: str) -> str:
    backoff = 2
    for attempt in range(6):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.output_text or ""
        except Exception:
            if attempt == 5:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    return ""


def sanitize_single_sentence(text: str) -> str:
    """
    Enforce: exactly one sentence returned.
    - take first non-empty line
    - strip wrapping quotes
    - remove accidental bullets/numbering
    """
    t = (text or "").strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = lines[0] if lines else ""

    t = t.strip().strip('"').strip("“”").strip("'").strip()
    t = re.sub(r"^\s*[\-\*\u2022]\s+", "", t)      # bullet
    t = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", t)     # 1. / 1)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def generate_foil_from_caption(caption: str, model: str) -> str:
    user_prompt = USER_PROMPT_TEMPLATE.format(caption=caption)
    raw = _call_openai(SYSTEM_PROMPT, user_prompt, model=model)
    foil = sanitize_single_sentence(raw)
    if not foil:
        raise ValueError(f"Empty foil generated. Raw output:\n{raw}")
    return foil

# =========================
# MULTIPLE CHOICE PROMPT UPDATE
# =========================
def replace_option_line(mc_prompt: str, option_letter: str, new_sentence: str) -> str:
    """
    Replace the sentence on the option line 'A. ...\n' or 'B. ...\n'
    Keeps 'A. ' / 'B. ' and replaces only the text until newline.
    """
    if not mc_prompt:
        return mc_prompt

    pat = re.compile(rf"(^|\n)(\s*{option_letter}\.\s*)([^\n]*)", re.MULTILINE)
    m = pat.search(mc_prompt)
    if not m:
        raise ValueError(
            f"Cannot find option '{option_letter}.' line in multiple_choice_prompt:\n{mc_prompt}"
        )

    start, end = m.span(3)  # the sentence part
    return mc_prompt[:start] + new_sentence + mc_prompt[end:]


def update_mc_prompt(mc_prompt: str, caption_position: int, foil_sentence: str) -> str:
    """
    Your dataset semantics:
    - caption_position == 0 => caption is in B => A is the foil => replace A
    - caption_position == 1 => caption is in A => B is the foil => replace B
    """
    if caption_position == 0:
        return replace_option_line(mc_prompt, "A", foil_sentence)
    elif caption_position == 1:
        return replace_option_line(mc_prompt, "B", foil_sentence)
    else:
        raise ValueError(f"caption_position must be 0 or 1, got: {caption_position}")

# =========================
# IO JSON (array)
# =========================
def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON array (list) in {path}, got {type(obj)}")
    return obj


def save_json_array(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Generate new foils from caption and update JSON array.")
    parser.add_argument("--in_json", type=str, required=True, help="Input JSON path (array of objects)")
    parser.add_argument("--out_json", type=str, required=True, help="Output JSON path (array of objects)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--save_every", type=int, default=1, help="Checkpoint every N items (default: 100)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep seconds between API calls")
    args = parser.parse_args()

    items = load_json_array(args.in_json)
    print(f"📦 Loaded JSON: {args.in_json} | n={len(items)}")

    SAVE_EVERY = max(1, int(args.save_every))

    for idx in tqdm(range(len(items)), desc="Generating foils"):
        item = items[idx]

        if "caption" not in item:
            raise KeyError(f"Item {idx}: missing 'caption'")
        if "multiple_choice_prompt" not in item:
            raise KeyError(f"Item {idx}: missing 'multiple_choice_prompt'")
        if "caption_position" not in item:
            raise KeyError(f"Item {idx}: missing 'caption_position'")

        caption = str(item["caption"])
        cap_pos = int(item["caption_position"])

        new_foil = generate_foil_from_caption(caption, model=args.model)

        item["foil"] = new_foil
        item["multiple_choice_prompt"] = update_mc_prompt(
            mc_prompt=str(item["multiple_choice_prompt"]),
            caption_position=cap_pos,
            foil_sentence=new_foil,
        )

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

        if (idx + 1) % SAVE_EVERY == 0:
            save_json_array(args.out_json, items)
            print(f"💾 Checkpoint saved: {args.out_json} (items={idx+1})")

    save_json_array(args.out_json, items)
    print(f"✅ Done. Wrote {len(items)} items to: {args.out_json}")


if __name__ == "__main__":
    main()
