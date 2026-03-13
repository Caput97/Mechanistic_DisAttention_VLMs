import os
# Limit visible GPUs (optional, keep if needed for your cluster)
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"

#3 and 4 GPU are the best
#0 and 5 seem to work ok too

import sys
# Make sure local site-packages are visible (adapt to your env)
sys.path.insert(0, "/home/dtesta/miniconda3/envs/qwenVL/lib/python3.10/site-packages")

from setproctitle import setproctitle
setproctitle("Attn_VLM")

import re
from collections import defaultdict
import requests

import argparse
from tqdm import tqdm
import av

import torch



import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import json

from attn_knockout.patching import patch_all_attention_layers
from attn_knockout.tracking_prob import track_sentence_prob_layerwise
from attn_knockout.tracking_attn import track_attention_layerwise
from attn_knockout.utils import get_video_id
from attn_knockout.models import load_model


device = "cuda" if torch.cuda.is_available() else "cpu"
np.set_printoptions(threshold=10000)



# -----------------------------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------
    # Run mode from command-line
    # -----------------------------
    # Expected usage:
    #   python script.py prob --query_scope assistant_token / --query_scope assistant_content
    #   python script.py attention_weights --query_scope assistant_token / --query_scope assistant_content

    parser = argparse.ArgumentParser()
    parser.add_argument("run_mode", nargs="?", default="prob", choices=["prob", "attention_weights"])
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
    help="Select which video to use: per-item video from dataset ('video') or a fixed black video ('black_video')."
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
    
    parser.add_argument(
    "--attn_per_head",
    action="store_true",
    help="If set (only in attention_weights mode), also compute per-head attention mass and save to a separate JSONL."
)
    
    parser.add_argument(
    "--attn_per_head_only",
    action="store_true",
    help="If set, in attention_weights mode compute/save ONLY per-head stats (skip average/topk JSONL)."
)


    args = parser.parse_args()

    RUN_MODE = args.run_mode
    query_scope = args.query_scope
    video_mode = args.video_mode
    ATTN_PER_HEAD = args.attn_per_head or args.attn_per_head_only
    ATTN_PER_HEAD_ONLY = args.attn_per_head_only



    MASK_REGION_MAP = {
    # CLI name      # internal name used by build_key_spans_for_mode()
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

        # comma-separated list
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out = []
        for p in parts:
            if p not in MASK_REGION_MAP:
                raise ValueError(
                    f"Unknown --mask_regions value '{p}'. "
                    f"Valid: all_regions, none, {', '.join(MASK_REGION_MAP.keys())}"
                )
            out.append(MASK_REGION_MAP[p])

        # remove duplicates while preserving order
        seen = set()
        out_unique = []
        for m in out:
            if m not in seen:
                out_unique.append(m)
                seen.add(m)

        return out_unique


    mask_modes = parse_mask_regions(args.mask_regions)
    mask_modes_tag = "__".join(mask_modes)
    

    if RUN_MODE == "attention_weights":
        if ATTN_PER_HEAD_ONLY:
            attn_mode_str = "per_head_ONLY"
        elif ATTN_PER_HEAD:
            attn_mode_str = "average_&_per_head"
        else:
            attn_mode_str = "average_only"
    else:
        attn_mode_str = "N/A"




    print(f"✅ RUN_MODE={RUN_MODE} | query_scope={query_scope} | video_mode={video_mode} | ✅ mask_modes={mask_modes} | attention_mode={attn_mode_str}")



    # -----------------------------
    # Paths and configuration
    # -----------------------------

    #mc_data = '/home/dtesta/XAI-VLMs-MC_consistency/MAIA_MC_task_SpazialeTotale.json'  
    #mc_data = "/home/dtesta/XAI-VLMs-MC_consistency/MAIA_MC_task_TemporaleParziale.json"
    #mc_data = "/home/dtesta/VLM_interpretability/MAIA_MC_task_TemporaleParziale_itempool1.json"
    #mc_data = "/home/dtesta/VLM_interpretability/MAIA_MC_task_SpazialeParziale_itempool1.json"
    #mc_data = "/home/dtesta/Mechanistic_DisAttention_VLMs/data/MAIA_MC_task_Spatial_itempool1_100_eng.json"
    #mc_data = "/home/dtesta/Mechanistic_DisAttention_VLMs/data/MAIA_MC_task_TemporaleParziale_itempool1_100_eng.json"
    #mc_data = "/home/dtesta/Mechanistic_DisAttention_VLMs/data/MAIA_MC_task_Causale_itempool1.json"
    #mc_data = "/home/dtesta/Mechanistic_DisAttention_VLMs/data/MAIA_MC_task_SpazialeParziale_itempool1_fakeFoils.json"
    #mc_data = "/home/dtesta/Mechanistic_DisAttention_VLMs/data/MAIA_MC_task_priorCheck_itempool1.json"
    mc_data = "/home/dtesta/Mechanistic_DisAttention_VLMs/data/winoground/winoground_caption_foil_mc.json"
    #subdataset = "spatial"
    #subdataset = "temporal"
    #subdataset = "causal"
    #subdataset = "fakeFoils"
    subdataset = "img"

    #video_folder = "/home/dtesta/MAIA_def/dataset_def/videos"
    video_folder = "/home/dtesta/Mechanistic_DisAttention_VLMs/data/winoground/img_winoground"
    black_video_path = "/home/dtesta/MAIA_def/dataset_def/videos/black_video.mp4"

    if video_mode == "black_video" and not os.path.exists(black_video_path):
        raise FileNotFoundError(f"black_video not found: {black_video_path}")



    # Output file for sentence probabilities
    output_path_prob = (f"/home/dtesta/Mechanistic_DisAttention_VLMs/results_knockout/qwen2.5vl/Attn_knockout_100_16F_{query_scope}_{subdataset}_eng_{video_mode}_mask_{mask_modes_tag}.jsonl")
    #output_path_prob = (f"/home/dtesta/Mechanistic_DisAttention_VLMs/results_knockout/qwen2.5vl/Attn_knockout_40_16F_ass_cont_priors_ita.jsonl")

    # Output file for attention weights (token-wise + top-k)
    output_path_attn = (f"/home/dtesta/Mechanistic_DisAttention_VLMs/results_knockout/qwen2.5vl/Attn_weights_100_16F_{query_scope}_{subdataset}_eng_{video_mode}_mask_{mask_modes_tag}_{attn_mode_str}.jsonl")
    #output_path_attn = (f"/home/dtesta/Mechanistic_DisAttention_VLMs/results_knockout/qwen2.5vl/Attn_weights_40_16F_ass_cont_priors_ita.jsonl")

    output_path_attn_per_head = (
    f"/home/dtesta/Mechanistic_DisAttention_VLMs/results_knockout/qwen2.5vl/"
    f"attn_weights_per_head_100_16F_{query_scope}_{subdataset}_eng_{video_mode}_mask_{mask_modes_tag}_{attn_mode_str}.jsonl"
)  
    #output_path_attn_per_head = (
    #f"/home/dtesta/Mechanistic_DisAttention_VLMs/results_knockout/qwen2.5vl/"
    #f"attn_weights_per_head__40_16F_ass_cont_priors_ita.jsonl"
#)


    # Directory to store full attention matrices as .pt files (one per item)
    #attn_pt_dir = f"/home/dtesta/Mechanistic_DisAttention_VLMs/results_knockout/qwen2.5vl/attn_matrices_pt_{query_scope}_{subdataset}_{video_mode}_mask_{mask_modes_tag}_{attn_mode_str}"
    
    os.makedirs(os.path.dirname(output_path_prob), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_attn), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_attn_per_head), exist_ok=True)
    #os.makedirs(attn_pt_dir, exist_ok=True)


    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    # -----------------------------
    # Load data
    # -----------------------------
    with open(mc_data, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("✅ JSON file loaded.")

    # Clean "multiple_choice_prompt" fields
    for item in data:
        s = item["multiple_choice_prompt"]
        # remove "Rispondi solo A o B.\n" if is at the end
        s_clean = re.sub(r"Rispondi solo A o B\.\n$", "", s)
        item["multiple_choice_prompt"] = s_clean


    # -----------------------------
    # Load model and patch attention
    # -----------------------------
    model, processor, tokenizer, lm_head = load_model(model_id)
    model.eval()

    print(type(model.model.layers[0].self_attn))
    print(model.model.layers[0].self_attn.forward)
    print()
    print() 

    #APPLY the monkey patch
    # This patches the forward method of Qwen2_5_VLAttention for all instances of Qwen2_5_VLAttention
    patch_all_attention_layers(model)

    
    print(f"Starting tracking (mode={RUN_MODE})...\n")

    #all_results_layer = []
    failed_videos = []

    # Example: process only a slice of the dataset (here item 6)
    #for idx, item in enumerate(tqdm(data[6:7], desc="Processing items"), start=6):
    for idx, item in enumerate(tqdm(data[:100], desc="Processing items")):

        try:
            prompt = item["multiple_choice_prompt"]
            video_id = item["video_id"]
            if video_mode == "black_video":
                video_path = black_video_path
            else:
                
                #da rimettere se uso i video
                #number = get_video_id(video_id)
                #video_path = os.path.join(video_folder, f"{number}.mp4")

                #solo per img
                video_path = os.path.join(video_folder, f"{video_id}.jpg")


            caption = item["caption"]
            foil = item["foil"]
            target = item["caption_position"] # 0 or 1 --> 0 means caption is B, 1 means caption is A 

            #mask_modes = ["none", "user", "vision", "vision_half1", "vision_half2", "user_text", "A_content", "B_content" ]
            

            if RUN_MODE == "prob":
                # ----------------------------------------
                # Sentence probability tracking mode
                # ----------------------------------------
                final_results = {
                    "caption": {},
                    "foil": {}
                }

                for mode in mask_modes:
                    prob_track = track_sentence_prob_layerwise(
                        prompt,
                        video_path,
                        caption,
                        foil,
                        model,
                        processor,
                        tokenizer,
                        lm_head,
                        mask_mode=mode,
                        query_scope=query_scope,
                    )

                    # prob_track structure: {"caption": {...layers...}, "foil": {...layers...}}
                    for label in ["caption", "foil"]:
                        for layer_key, scores in prob_track[label].items():
                            if layer_key not in final_results[label]:
                                final_results[label][layer_key] = {}
                            final_results[label][layer_key][mode] = scores

                item_result = {
                    "item": idx + 1,
                    "video_id": video_id,
                    "target": target,
                    "results": final_results
                }

                with open(output_path_prob, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item_result, ensure_ascii=False) + "\n")

                print(f"✅ item {idx+1} saved (prob)")


            elif RUN_MODE == "attention_weights":
                # ----------------------------------------
                # Attention weights tracking mode
                #   - Save full attn matrices per item in a .pt file
                #   - Save only topk, target_tokens and region_stats in JSONL
                # ----------------------------------------

                # average outputs (only if not per-head-only)
                final_results_attn = {"caption": {}, "foil": {}} if not ATTN_PER_HEAD_ONLY else None
                attn_mats_item     = {"caption": {}, "foil": {}} if not ATTN_PER_HEAD_ONLY else None

                # per-head outputs (only if ATTN_PER_HEAD is True)
                final_results_attn_per_head = {"caption": {}, "foil": {}} if ATTN_PER_HEAD else None



                for mode in mask_modes:
                    attn_track = track_attention_layerwise(
                        prompt,
                        video_path,
                        caption,
                        foil,
                        model,
                        processor,
                        tokenizer,
                        mask_mode=mode,
                        query_scope=query_scope,
                        return_per_head=ATTN_PER_HEAD
                    )

                    # attn_track structure:
                    # {
                    #   "caption": { layer_k: { "attn_tensor", "target_tokens", "topk", "region_stats" }, ... },
                    #   "foil":    { ... }
                    # }

                    for label in ["caption", "foil"]:
                        if label not in attn_track:
                            continue

                        for layer_key, layer_content in attn_track[label].items():
                            # 1) Save average summaries (only if not per-head-only)
                            if not ATTN_PER_HEAD_ONLY:
                                # 1) Store full attn tensor for .pt saving
                                #if layer_key not in attn_mats_item[label]:
                                #    attn_mats_item[label][layer_key] = {}
                                #attn_mats_item[label][layer_key][mode] = layer_content["attn_tensor"]

                                # 2) Store only lightweight summaries in JSON
                                if layer_key not in final_results_attn[label]:
                                    final_results_attn[label][layer_key] = {}

                                final_results_attn[label][layer_key][mode] = {
                                    "target_tokens": layer_content["target_tokens"],
                                    "query_rows_used": layer_content["query_rows_used"],
                                    "topk": layer_content["topk"],
                                    "region_stats": layer_content["region_stats"],
                                }

                            # 3) Store per-head results (optional)
                            if ATTN_PER_HEAD and "per_head" in layer_content:
                                if layer_key not in final_results_attn_per_head[label]:
                                    final_results_attn_per_head[label][layer_key] = {}
                                final_results_attn_per_head[label][layer_key][mode] = layer_content["per_head"]


                # Save all attention matrices for this item as a single .pt file
                #attn_pt_path = os.path.join(
                #    attn_pt_dir,
                #    f"item_{idx+1}_attn_matrices.pt"
                #)
                #torch.save(attn_mats_item, attn_pt_path)

                # 4) Save average-head JSONL
                # JSONL record with summaries + reference to the .pt file
                # Save average JSONL (only if not per-head-only)
                if not ATTN_PER_HEAD_ONLY:
                    item_attn_result = {
                        "item": idx + 1,
                        "video_id": video_id,
                        "target": target,
                        #"attn_file": attn_pt_path,
                        "results": final_results_attn
                    }

                    with open(output_path_attn, "a", encoding="utf-8") as f:
                        f.write(json.dumps(item_attn_result, ensure_ascii=False) + "\n")

                    print(f"✅ item {idx+1} saved (attention_weights)")

                # 5) Save per-head JSONL (only if enabled)
                if ATTN_PER_HEAD or ATTN_PER_HEAD_ONLY:
                    item_attn_per_head = {
                        "item": idx + 1,
                        "video_id": video_id,
                        "target": target,
                        "results": final_results_attn_per_head
                    }
                    with open(output_path_attn_per_head, "a", encoding="utf-8") as f:
                        f.write(json.dumps(item_attn_per_head, ensure_ascii=False) + "\n")
                
                    print(f"✅ item {idx+1} saved (attention_weights per head)")


        except FileNotFoundError as e:
            print()
            print(f"[ERROR] Failed saving (mode={RUN_MODE})")
            print(f"Reason: {e}")
            print()
            continue

        except av.error.FFmpegError as e:
            msg = str(e)
            if "NAL unit" in msg or "partial file" in msg or "Error splitting the input into NAL units" in msg:
                print()
                print(f"[ERROR] Video decode/corrupt on {video_id} (path: {video_path}) - entry {idx}")
                print(f"Reason: {e}")
                print()
                failed_videos.append((idx, video_id))
                continue
            else:
                raise

        except Exception as e:
            print()
            print(f"[ERROR] Failed on video {video_id} (path: {video_path}) - entry {idx}")
            print(f"Reason: {e}")
            print()
            failed_videos.append((idx, video_id))
            continue
        
        finally:
            # clean GPU memory after each 5 items
            if idx % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    if RUN_MODE == "prob":
        print(f"✅ Layer-wise probability results saved to: {output_path_prob}")
    elif RUN_MODE == "attention_weights" and not ATTN_PER_HEAD_ONLY:
        print(f"✅ Layer-wise attention results saved to: {output_path_attn}")
    elif RUN_MODE == "attention_weights" and ATTN_PER_HEAD_ONLY:
        print(f"✅ Layer-wise attention results (per head) saved to: {output_path_attn_per_head}")
    elif RUN_MODE == "attention_weights" and ATTN_PER_HEAD:
        print(f"✅ Layer-wise attention results saved to: {output_path_attn}")
        print(f"✅ Layer-wise attention results (per head) saved to: {output_path_attn_per_head}")

    if failed_videos:
        print("⚠ The following items failed:")
        for idx, vid in failed_videos:
            print(f"  - index {idx}, video_id={vid}")


