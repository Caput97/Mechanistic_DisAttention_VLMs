#!/usr/bin/env python3
import os
import json
import argparse
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

REGIONS = ["video", "user_text", "A", "B", "other"]
LABELS = ["caption", "foil"]


def parse_layer_idx(layer_key: str) -> int:
    """Parse integer layer index from keys like 'layer_0', 'layer_12', etc."""
    if not layer_key.startswith("layer_"):
        raise ValueError(f"Unexpected layer key: {layer_key}")
    return int(layer_key.split("_")[1])


def safe_filename(s: str) -> str:
    """Make a string safe to use in filenames."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def collect_values(jsonl_path: str):
    """
    Collect region_stats['global'] values for all items.
    We store lists of floats for each (label, mask_mode, layer_idx, region).
    """
    values = defaultdict(list)  # key: (label, mask_mode, layer_idx, region) -> [v,...]
    layer_set = set()
    mask_set = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            results = obj.get("results", {})

            for label in LABELS:
                if label not in results:
                    continue
                label_block = results[label]  # layer_key -> mask_mode -> payload

                for layer_key, layer_content in label_block.items():
                    try:
                        layer_idx = parse_layer_idx(layer_key)
                    except Exception:
                        continue

                    layer_set.add(layer_idx)

                    for mask_mode, payload in layer_content.items():
                        mask_set.add(mask_mode)
                        global_stats = payload.get("region_stats", {}).get("global", {})

                        for region in REGIONS:
                            v = global_stats.get(region, None)
                            if v is None:
                                continue
                            values[(label, mask_mode, layer_idx, region)].append(float(v))

    layers = sorted(layer_set)
    masks = sorted(mask_set)
    return values, layers, masks


def compute_mean_std_counts(values, layers, masks):
    """
    Compute mean/std/count for each (label, mask_mode, layer, region).
    """
    mean = {label: {m: {li: {r: np.nan for r in REGIONS} for li in layers} for m in masks} for label in LABELS}
    std  = {label: {m: {li: {r: np.nan for r in REGIONS} for li in layers} for m in masks} for label in LABELS}
    cnt  = {label: {m: {li: {r: 0 for r in REGIONS} for li in layers} for m in masks} for label in LABELS}

    for label in LABELS:
        for m in masks:
            for li in layers:
                for r in REGIONS:
                    lst = values.get((label, m, li, r), [])
                    cnt[label][m][li][r] = len(lst)
                    if len(lst) == 0:
                        continue
                    arr = np.array(lst, dtype=np.float32)
                    mean[label][m][li][r] = float(np.mean(arr))
                    std[label][m][li][r]  = float(np.std(arr, ddof=0))

    return mean, std, cnt


def plot_caption_vs_foil_per_mask_region(
    layers: List[int],
    masks: List[str],
    mean: Dict[str, Any],
    std: Dict[str, Any],
    out_dir: str,
    with_std_band: bool = True,
):
    """
    Create plots:
      for each mask_mode and region:
        x = layer
        y = mean global attention mass
        lines: caption vs foil
        optional: +-1 std band
    """
    os.makedirs(out_dir, exist_ok=True)

    for m in masks:
        for region in REGIONS:
            plt.figure(figsize=(10, 4.5))

            for label in LABELS:
                y = np.array([mean[label][m][li][region] for li in layers], dtype=np.float32)
                plt.plot(layers, y, marker="o", linewidth=1, label=label)

                if with_std_band:
                    s = np.array([std[label][m][li][region] for li in layers], dtype=np.float32)
                    plt.fill_between(layers, y - s, y + s, alpha=0.15)

            plt.ylim(0, 1)
            plt.xlabel("Layer")
            plt.ylabel("Mean global attention mass (normalized)")
            plt.title(f"Dataset mean | mode={m} | region={region} | caption vs foil")
            plt.xticks(layers)
            plt.legend()
            plt.tight_layout()

            fname = safe_filename(f"DATASET_MEAN_mode_{m}_region_{region}_caption_vs_foil.png")
            plt.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.close()


def plot_summary_per_mask_all_regions(
    layers: List[int],
    masks: List[str],
    mean: Dict[str, Any],
    std: Dict[str, Any],
    out_dir: str,
    with_std_band: bool = False,
):
    """
    Optional compact plots:
      for each mask_mode:
        one plot with multiple lines = regions
        and caption/foil separated by linestyle (still readable enough).
    """
    os.makedirs(out_dir, exist_ok=True)

    for m in masks:
        plt.figure(figsize=(10, 4.5))

        # Caption lines
        for region in REGIONS:
            y = np.array([mean["caption"][m][li][region] for li in layers], dtype=np.float32)
            plt.plot(layers, y, marker="o", linewidth=1, label=f"caption:{region}")

        # Foil lines
        for region in REGIONS:
            y = np.array([mean["foil"][m][li][region] for li in layers], dtype=np.float32)
            plt.plot(layers, y, marker="x", linewidth=1, label=f"foil:{region}")

        plt.ylim(0, 1)
        plt.xlabel("Layer")
        plt.ylabel("Mean global attention mass (normalized)")
        plt.title(f"Dataset mean | mode={m} | all regions | caption vs foil (compact)")
        plt.xticks(layers)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()

        fname = safe_filename(f"DATASET_MEAN_mode_{m}_ALLREGIONS_compact.png")
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot dataset-mean region_stats['global'] over layers, caption vs foil, per masking setting."
    )
    parser.add_argument("--jsonl", required=True, help="Path to attention summaries JSONL")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    parser.add_argument("--no_std", action="store_true", help="Disable +-1 std bands")
    parser.add_argument("--compact", action="store_true", help="Also produce compact plots (all regions in one figure per mask_mode)")
    args = parser.parse_args()

    values, layers, masks = collect_values(args.jsonl)
    mean, std, cnt = compute_mean_std_counts(values, layers, masks)

    # Main: most interpretable set of plots (mask_mode x region)
    plot_caption_vs_foil_per_mask_region(
        layers, masks, mean, std,
        out_dir=args.out_dir,
        with_std_band=(not args.no_std),
    )

    # Optional: compact plots to reduce number of figures
    if args.compact:
        plot_summary_per_mask_all_regions(
            layers, masks, mean, std,
            out_dir=args.out_dir,
            with_std_band=False,
        )

    print(f"[OK] Saved plots to: {args.out_dir}")
    print(f"[INFO] Layers found: {layers}")
    print(f"[INFO] Mask modes found: {masks}")


if __name__ == "__main__":
    main()
