import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"
import sys
sys.path.insert(0, "/home/dtesta/miniconda3/envs/qwenVL/lib/python3.10/site-packages")

from setproctitle import setproctitle
setproctitle("Attn_VLM_AB")

import re
import argparse
from tqdm import tqdm
import av
import torch
import numpy as np
import json

from attn_knockout.patching import patch_all_attention_layers
from attn_knockout.utils import get_video_id, track_ab_prob_layerwise
from attn_knockout.models import load_model


device = "cuda" if torch.cuda.is_available() else "cpu"
np.set_printoptions(threshold=10000)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_scope",
        type=str,
        default="assistant_token",
        choices=["assistant_token", "assistant_content"],
        help="Which assistant query rows receive the masking."
    )

    parser.add_argument(
        "--video_mode",
        type=str,
        default="video",
        choices=["video", "black_video"],
        help="Use per-item video ('video') or a fixed black video ('black_video')."
    )

    parser.add_argument(
        "--mask_regions",
        type=str,
        default="all_regions",
        help=(
            "Which regions to mask. "
            "Use 'all_regions' for the default set, "
            "'none' for baseline only, "
            "or a comma-separated list: user_role,user_text,video,A_content,B_content,video1half,video2half"
        ),
    )

    args = parser.parse_args()

    query_scope = args.query_scope
    video_mode = args.video_mode

    MASK_REGION_MAP = {
        "none": "none",
        "user_role": "user",
        "user_text": "user_text",
        "video": "vision",
        "A_content": "A_content",
        "B_content": "B_content",
        "video1half": "vision_half1",
        "video2half": "vision_half2",
    }

    DEFAULT_MASK_MODES = [
        "none", "user", "vision", "vision_half1", "vision_half2",
        "user_text", "A_content", "B_content"
    ]

    def parse_mask_regions(mask_regions_str: str):
        s = (mask_regions_str or "").strip()
        if not s or s == "all_regions":
            return DEFAULT_MASK_MODES
        if s == "none":
            return ["none"]

        parts = [p.strip() for p in s.split(",") if p.strip()]
        out = []
        for p in parts:
            if p not in MASK_REGION_MAP:
                raise ValueError(
                    f"Unknown --mask_regions value '{p}'. "
                    f"Valid: all_regions, none, {', '.join(MASK_REGION_MAP.keys())}"
                )
            out.append(MASK_REGION_MAP[p])

        seen = set()
        out_unique = []
        for m in out:
            if m not in seen:
                out_unique.append(m)
                seen.add(m)
        return out_unique

    mask_modes = parse_mask_regions(args.mask_regions)
    mask_modes_tag = "__".join(mask_modes)

    print(f"✅ RUN_MODE=ab_prob | query_scope={query_scope} | video_mode={video_mode} | ✅ mask_modes={mask_modes}")

    # -----------------------------
    # Paths and configuration
    # -----------------------------
    #mc_data = "/home/dtesta/Mechanistic_DisAttention_VLMs/data/MAIA_MC_task_TemporaleParziale_itempool1.json"
    mc_data = "/home/dtesta/Mechanistic_DisAttention_VLMs/data/MAIA_MC_task_priorCheck_itempool1.json"
    #subdataset = "temporal"
    subdataset = "priors"

    video_folder = "/home/dtesta/MAIA_def/dataset_def/videos"
    black_video_path = "/home/dtesta/MAIA_def/dataset_def/videos/black_video.mp4"

    if video_mode == "black_video" and not os.path.exists(black_video_path):
        raise FileNotFoundError(f"black_video not found: {black_video_path}")

    output_path_ab = (
        f"/home/dtesta/Mechanistic_DisAttention_VLMs/results_knockout/qwen2.5vl/gen_AB/"
        f"AB_prob_40_16F_{query_scope}_{subdataset}_{video_mode}_mask_{mask_modes_tag}.jsonl"
    )
    os.makedirs(os.path.dirname(output_path_ab), exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    with open(mc_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("✅ JSON file loaded.")

    # (opzionale) come nel tuo run.py: togli la vecchia riga finale se presente,
    # poi la ricostruiamo in modo standard dentro track_ab_prob_layerwise().
    for item in data:
        s = item["multiple_choice_prompt"]
        s_clean = re.sub(r"Rispondi solo A o B\.\n$", "", s)
        item["multiple_choice_prompt"] = s_clean

    # -----------------------------
    # Load model and patch attention
    # -----------------------------
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model, processor, tokenizer, lm_head = load_model(model_id)
    model.eval()

    patch_all_attention_layers(model)

    failed_videos = []

    for idx, item in enumerate(tqdm(data[:40], desc="Processing items")):
        try:
            prompt = item["multiple_choice_prompt"]
            video_id = item["video_id"]

            if video_mode == "black_video":
                video_path = black_video_path
            else:
                number = get_video_id(video_id)
                video_path = os.path.join(video_folder, f"{number}.mp4")

            target = item["caption_position"]  # 0 -> caption=B, 1 -> caption=A (come nel tuo)

            final_results = {}  # mode -> layer_k -> stats

            for mode in mask_modes:
                layerwise = track_ab_prob_layerwise(
                    prompt=prompt,
                    video_path=video_path,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    lm_head=lm_head,
                    mask_mode=mode,
                    query_scope=query_scope,
                )
                final_results[mode] = layerwise

            item_result = {
                "item": idx + 1,
                "video_id": video_id,
                "target": target,
                "results": final_results
            }

            with open(output_path_ab, "a", encoding="utf-8") as f:
                f.write(json.dumps(item_result, ensure_ascii=False) + "\n")

        except av.error.FFmpegError as e:
            failed_videos.append((idx, item.get("video_id", "UNK"), str(e)))
            continue
        except Exception as e:
            failed_videos.append((idx, item.get("video_id", "UNK"), str(e)))
            continue
        finally:
            if idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    print(f"✅ AB layerwise results saved to: {output_path_ab}")
    if failed_videos:
        print("⚠ The following items failed:")
        for i, vid, err in failed_videos:
            print(f"  - index {i}, video_id={vid}, err={err}")
