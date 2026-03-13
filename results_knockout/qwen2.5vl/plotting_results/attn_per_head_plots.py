
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate comparable contrast heatmaps (video_mass - user_text_mass) from attention-per-head JSONL files.

For each input file:
- Output subfolder name extracted from filename stem:
  take the token immediately after 'attn_weights_per_head_200_16F_assistant_content_' and before next '_'
  e.g. '...content_spatial_video_mask...' -> 'spatial'
- Produce 6 heatmaps:
  - caption: none / vision / user_text
  - foil:    none / vision / user_text

Heatmap:
- background: mean contrast across items (layer x head)
- colormap: red ↔ blue (default RdBu_r), centered at 0, with shared global scale:
  vmax = pXX percentile of |contrast| across ALL input files (default p99)
- overlay markers (fixed size):
  - 'o' = top-3 video-oriented heads per layer (highest mean contrast)
  - 'x' = top-3 user_text-oriented heads per layer (lowest mean contrast)

Usage:
only with mean contrast heatmaps:
python make_attn_contrast_heatmaps.py \
  --input /path/to/attn/*.jsonl \
  --outdir /path/to/output_plots \
  --percentile 99 \
  --dpi 250

Optional (also generate STD heatmaps):
python make_attn_contrast_heatmaps.py \
  --input /path/to/attn/*.jsonl \
  --outdir /path/to/output_plots \
  --percentile 99 \
  --make-std \
  --std-cmap viridis \
  --dpi 250

  Optional (also generate combined mean+std heatmaps):
    In this heatmap, color encodes the direction of the modality bias 
    (video vs text), while opacity reflects the degree 
    of input-dependent variability.

  python attn_per_head_plots.py \
  --input file1.jsonl file2.jsonl \
  --outdir /path/to/plots \
  --percentile 99 \
  --make-std \
  --make-combined \
  --combined-modes routing grounding \
  --min-alpha 0.15 \
  --dpi 250

  here we can also decide to choose which combined variants to generate (also only one):
    --combined-modes routing grounding
  where:
    - routing   (Case A): alpha = 1 - std_norm  (stable => more opaque)
    - grounding (Case B): alpha = std_norm      (variable => more opaque    

"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


DEFAULT_PREFIX = "attn_weights_per_head_200_16F_assistant_content_"
DEFAULT_MODES = ["none", "vision", "user_text"]
DEFAULT_LABELS = ["caption", "foil"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot comparable contrast heatmaps (video - user_text) with global pXX scale across multiple JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", type=Path, nargs="+", required=True, help="One or more input .jsonl files")
    p.add_argument("--outdir", "-o", type=Path, required=True, help="Output directory (subfolders per file)")
    p.add_argument("--percentile", type=float, default=99.0, help="Global percentile for |contrast| to set vmax")
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="Mask modes to plot")
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, help="Labels to plot: caption and/or foil")
    p.add_argument("--cmap", type=str, default="RdBu_r", help="Matplotlib colormap (red↔blue recommended: RdBu_r)")
    p.add_argument("--dpi", type=int, default=200, help="DPI for saved images")
    p.add_argument("--marker-size", type=float, default=25.0, help="Fixed marker size (points^2) for X and O")
    p.add_argument("--linewidth", type=float, default=1.3, help="Marker linewidth")
    p.add_argument("--make-std", action="store_true", help="Also generate STD heatmaps")
    p.add_argument("--std-cmap", type=str, default="viridis", help="Colormap for STD heatmaps (sequential)")
    p.add_argument("--make-combined", action="store_true", help="Also generate combined mean+std heatmaps")
    p.add_argument(
        "--combined-modes",
        nargs="+",
        default=["routing", "grounding"],
        choices=["routing", "grounding"],
        help="Which combined variants to generate: routing (Case A) and/or grounding (Case B)",
    )
    p.add_argument("--min-alpha", type=float, default=0.15, help="Minimum alpha for combined heatmaps")
    p.add_argument("--alpha-cmap", type=str, default="Greys", help="Colormap used to show opacity colorbar")


    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No JSONL records found in {path}")
    return rows


def sorted_layer_keys(results_for_label: Dict) -> List[str]:
    # expects keys like "layer_0" ... "layer_27"
    def layer_num(k: str) -> int:
        try:
            return int(k.split("_")[1])
        except Exception:
            return 10**9

    keys = list(results_for_label.keys())
    keys.sort(key=layer_num)
    return keys


def infer_shape_and_layers(rows: List[Dict], labels: List[str], modes: List[str]) -> Tuple[List[str], int]:
    """
    Find layer_keys and number of heads H from the first record that contains them.
    """
    for r in rows:
        results = r.get("results", {})
        for label in labels:
            if label in results and isinstance(results[label], dict) and results[label]:
                layer_keys = sorted_layer_keys(results[label])
                for mode in modes:
                    try:
                        payload = results[label][layer_keys[0]][mode]
                        H = len(payload["contrast_video_vs_user_text"])
                        return layer_keys, H
                    except Exception:
                        continue
    raise RuntimeError("Could not infer layer keys / head count from input JSONL.")


def compute_mean_contrast(rows: List[Dict], label: str, mode: str, layer_keys: List[str], H: int) -> np.ndarray:
    """
    Returns mean contrast matrix shape (L, H).
    """
    N = len(rows)
    L = len(layer_keys)
    all_c = np.zeros((N, L, H), dtype=np.float32)

    for i, r in enumerate(rows):
        res_label = r.get("results", {}).get(label, {})
        for l, lk in enumerate(layer_keys):
            try:
                c = res_label[lk][mode]["contrast_video_vs_user_text"]
                all_c[i, l, :] = np.asarray(c, dtype=np.float32)
            except Exception:
                # if missing, it stays zeros for that item/layer
                pass

    return all_c.mean(axis=0)  # (L, H)

def compute_std_contrast(rows: List[Dict], label: str, mode: str, layer_keys: List[str], H: int) -> np.ndarray:
    """
    Returns std contrast matrix shape (L, H), computed across items.
    """
    N = len(rows)
    L = len(layer_keys)
    all_c = np.zeros((N, L, H), dtype=np.float32)

    for i, r in enumerate(rows):
        res_label = r.get("results", {}).get(label, {})
        for l, lk in enumerate(layer_keys):
            try:
                c = res_label[lk][mode]["contrast_video_vs_user_text"]
                all_c[i, l, :] = np.asarray(c, dtype=np.float32)
            except Exception:
                pass

    return all_c.std(axis=0)  # (L,H)



def top3_by_mean(mean_contrast: np.ndarray, kind: str) -> np.ndarray:
    """
    mean_contrast shape (L, H)
    kind: "video" -> 3 largest; "user_text" -> 3 smallest
    Returns (L, 3) head indices.
    """
    L, H = mean_contrast.shape
    out = np.zeros((L, 3), dtype=int)

    for l in range(L):
        vals = mean_contrast[l]
        if kind == "video":
            out[l] = np.argsort(vals)[-3:][::-1]
        else:
            out[l] = np.argsort(vals)[:3]
    return out


def extract_subfolder_name(stem: str, prefix: str = DEFAULT_PREFIX) -> str:
    """
    Extract the token immediately after prefix and before next underscore.
    If prefix not found, fallback to last underscore chunk.
    """
    if prefix in stem:
        rest = stem.split(prefix, 1)[1]
        return rest.split("_", 1)[0] if "_" in rest else rest
    parts = stem.split("_")
    return parts[-1] if parts else stem


def compute_global_vmax(all_inputs: List[Path], labels: List[str], modes: List[str], percentile: float) -> float:
    """
    Compute global vmax as pXX of |mean_contrast| across all files/labels/modes.
    """
    abs_vals: List[float] = []

    for path in all_inputs:
        rows = load_jsonl(path)
        layer_keys, H = infer_shape_and_layers(rows, labels, modes)

        for label in labels:
            for mode in modes:
                mean_c = compute_mean_contrast(rows, label, mode, layer_keys, H)  # (L,H)
                abs_vals.append(np.abs(mean_c).ravel())

    abs_vals = np.concatenate(abs_vals, axis=0)
    vmax = float(np.percentile(abs_vals, percentile))
    # avoid degenerate scale
    if vmax <= 0:
        vmax = float(np.max(abs_vals)) if abs_vals.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
    return vmax

def compute_global_std_vmax(all_inputs: List[Path], labels: List[str], modes: List[str], percentile: float) -> float:
    """
    Compute global vmax for std as pXX of std_contrast across all files/labels/modes.
    """
    vals: List[np.ndarray] = []

    for path in all_inputs:
        rows = load_jsonl(path)
        layer_keys, H = infer_shape_and_layers(rows, labels, modes)

        for label in labels:
            for mode in modes:
                std_c = compute_std_contrast(rows, label, mode, layer_keys, H)  # (L,H)
                vals.append(std_c.ravel())

    vals = np.concatenate(vals, axis=0)
    vmax = float(np.percentile(vals, percentile))
    if vmax <= 0:
        vmax = float(np.max(vals)) if vals.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
    return vmax


def plot_heatmap(
    mean_contrast: np.ndarray,
    top_video: np.ndarray,
    top_text: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str,
    vmax: float,
    marker_size: float,
    linewidth: float,
    dpi: int,
):
    """
    mean_contrast: (L,H)
    top_video/top_text: (L,3) head indices
    """
    L, H = mean_contrast.shape
    img = mean_contrast.T  # (H, L)

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Head index")
    ax.set_title(title)

    # Colorbar con valori numerici + etichette semantiche
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean contrast (video_mass − user_text_mass)")
    cbar.ax.tick_params(labelsize=9)

    cbar.ax.text(
        0.5, 1.02, "video-oriented",
        ha="center", va="bottom",
        transform=cbar.ax.transAxes, fontsize=9
    )
    cbar.ax.text(
        0.5, -0.06, "text-oriented",
        ha="center", va="top",
        transform=cbar.ax.transAxes, fontsize=9
    )

    # Overlay X e O (3 + 3 per layer)
    for l in range(L):
        ax.scatter(
            np.full(3, l),
            top_video[l],
            s=marker_size,
            facecolors="none",
            edgecolors="black",
            marker="o",
            linewidths=linewidth,
            zorder=3,
        )
        ax.scatter(
            np.full(3, l),
            top_text[l],
            s=marker_size,
            c="black",
            marker="x",
            linewidths=linewidth,
            zorder=3,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_std_heatmap(
    std_contrast: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str,
    vmax: float,
    dpi: int,
):
    
    #std_contrast: (L,H)
    
    L, H = std_contrast.shape
    img = std_contrast.T  # (H,L)

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Head index")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Std of contrast across items")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    

def plot_combined_heatmap(
    mean_contrast: np.ndarray,   # (L,H)
    std_contrast: np.ndarray,    # (L,H)
    title: str,
    out_path: Path,
    cmap: str,
    mean_vmax: float,
    std_vmax: float,
    min_alpha: float,
    dpi: int,
    variant: str,                # "routing" (Case A) or "grounding" (Case B)
    alpha_cmap: str = "Greys",
):
    """
    Combined heatmap:
      - RGB encodes mean contrast with diverging cmap (centered at 0)
      - Alpha encodes a function of normalized std

    Variants:
      - routing   (Case A): alpha = 1 - std_norm  (stable => more opaque)
      - grounding (Case B): alpha = std_norm      (variable => more opaque)

    Adds two colorbars:
      1) Mean contrast (color) with video/text oriented labels
      2) Opacity mapping with input-invariant/input-dependent labels
    """
    L, H = mean_contrast.shape
    mean_img = mean_contrast.T  # (H,L)
    std_img  = std_contrast.T   # (H,L)

    # --- Mean -> RGBA (color)
    norm_mean = plt.Normalize(vmin=-mean_vmax, vmax=mean_vmax)
    cm_mean = plt.get_cmap(cmap)
    rgba = cm_mean(norm_mean(mean_img))  # (H,L,4)

    # --- Std -> normalized in [0,1]
    std_norm = np.clip(std_img / (std_vmax + 1e-12), 0.0, 1.0)

    if variant == "routing":
        # Case A: stable (low std) => opaque
        alpha = 1.0 - std_norm
        top_label = "input-invariant"
        bottom_label = "input-dependent"
        alpha_label = "Opacity = 1 − std/std_vmax"
    elif variant == "grounding":
        # Case B: variable (high std) => opaque
        alpha = std_norm
        top_label = "input-dependent"
        bottom_label = "input-invariant"
        alpha_label = "Opacity = std/std_vmax"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # keep cells faintly visible
    alpha = np.clip(alpha, min_alpha, 1.0)
    rgba[..., 3] = alpha

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(rgba, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head index")
    ax.set_title(title)

    # --- Two colorbars (to the right), well separated
    divider = make_axes_locatable(ax)
    cax_mean  = divider.append_axes("right", size="3.5%", pad=0.08)
    cax_alpha = divider.append_axes("right", size="3.5%", pad=0.75)

    # Colorbar #1: MEAN
    mappable_mean = plt.cm.ScalarMappable(norm=norm_mean, cmap=cm_mean)
    mappable_mean.set_array([])
    cbar1 = fig.colorbar(mappable_mean, cax=cax_mean)
    cbar1.set_label("Mean contrast (video_mass − user_text_mass)")
    cbar1.ax.tick_params(labelsize=9)
    cbar1.ax.text(0.5, 1.02, "video-oriented", ha="center", va="bottom",
                  transform=cbar1.ax.transAxes, fontsize=9)
    cbar1.ax.text(0.5, -0.06, "text-oriented", ha="center", va="top",
                  transform=cbar1.ax.transAxes, fontsize=9)

    # Colorbar #2: OPACITY mapping (shown with a grayscale legend)
    norm_alpha = plt.Normalize(vmin=0.0, vmax=1.0)
    cm_alpha = plt.get_cmap(alpha_cmap)
    mappable_alpha = plt.cm.ScalarMappable(norm=norm_alpha, cmap=cm_alpha)
    mappable_alpha.set_array([])
    cbar2 = fig.colorbar(mappable_alpha, cax=cax_alpha)
    cbar2.set_label(alpha_label)
    cbar2.ax.tick_params(labelsize=9)

    cbar2.ax.text(0.5, 1.02, top_label, ha="center", va="bottom",
                  transform=cbar2.ax.transAxes, fontsize=9)
    cbar2.ax.text(0.5, -0.06, bottom_label, ha="center", va="top",
                  transform=cbar2.ax.transAxes, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


   





def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Global scales (comparable across all input files)
    # --------------------------------------------------
    # Mean contrast (diverging colormap, centered at 0)
    vmax = compute_global_vmax(
        all_inputs=args.input,
        labels=args.labels,
        modes=args.modes,
        percentile=args.percentile,
    )
    print(
        f"[INFO] Global mean vmax = p{args.percentile:.1f}(|mean_contrast|) "
        f"across all inputs = {vmax:.6f}"
    )

    # Std contrast (sequential colormap, >= 0)
    std_vmax = None
    if args.make_std:
        std_vmax = compute_global_std_vmax(
            all_inputs=args.input,
            labels=args.labels,
            modes=args.modes,
            percentile=args.percentile,
        )
        print(
            f"[INFO] Global std vmax = p{args.percentile:.1f}(std_contrast) "
            f"across all inputs = {std_vmax:.6f}"
        )

    # --------------------------------------------------
    # Per-file processing
    # --------------------------------------------------
    for in_path in args.input:
        rows = load_jsonl(in_path)
        layer_keys, H = infer_shape_and_layers(rows, args.labels, args.modes)

        stem = in_path.stem
        tag = extract_subfolder_name(stem, prefix=DEFAULT_PREFIX)

        file_outdir = args.outdir / tag
        file_outdir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # >>> BLOCCO RICHIESTO (SEMPRE PRESENTE) <<<
        # --------------------------------------------------
        mean_dir = file_outdir / "mean"
        std_dir  = file_outdir / "std"
        mean_dir.mkdir(exist_ok=True)
        std_dir.mkdir(exist_ok=True)
        # --------------------------------------------------

        for label in args.labels:   # e.g. ["caption", "foil"]
            for mode in args.modes: # e.g. ["none", "vision", "user_text"]

                # ------------------------------------------
                # MEAN heatmap (with X/O markers)
                # ------------------------------------------
                mean_c = compute_mean_contrast(
                    rows, label, mode, layer_keys, H
                )  # shape (L, H)

                top_video = top3_by_mean(mean_c, kind="video")       # (L, 3)
                top_text  = top3_by_mean(mean_c, kind="user_text")   # (L, 3)

                mean_title = (
                    f"{tag} | {label} | mode={mode} | "
                    f"Attn_per-head layerwise"
                )
                mean_png = mean_dir / f"{tag}__{label}__{mode}__heatmap.png"

                plot_heatmap(
                    mean_contrast=mean_c,
                    top_video=top_video,
                    top_text=top_text,
                    title=mean_title,
                    out_path=mean_png,
                    cmap=args.cmap,
                    vmax=vmax,
                    marker_size=args.marker_size,
                    linewidth=args.linewidth,
                    dpi=args.dpi,
                )
                print(f"Saved: {mean_png}")

                # ------------------------------------------
                # STD heatmap (optional, no markers)
                # + reuse std_c for combined
                # ------------------------------------------
                std_c = None
                if args.make_std or args.make_combined:
                    std_c = compute_std_contrast(
                        rows, label, mode, layer_keys, H
                    )  # shape (L, H)

                    #check
                    std_norm = np.clip(std_c / (std_vmax + 1e-12), 0.0, 1.0)
                    print(tag, label, mode,
                        "std_vmax=", std_vmax,
                        "std min/median/p90/p99/max=",
                        float(std_c.min()),
                        float(np.median(std_c)),
                        float(np.percentile(std_c, 90)),
                        float(np.percentile(std_c, 99)),
                        float(std_c.max()),
                        "std_norm mean=", float(std_norm.mean()),
                        "std_norm p50/p90/p99=",
                        float(np.percentile(std_norm, 50)),
                        float(np.percentile(std_norm, 90)),
                        float(np.percentile(std_norm, 99))
                    )


                if args.make_std:
                    std_title = (
                        f"{tag} | {label} | mode={mode} | "
                        f"Attn_per-head layerwise (STD)"
                    )
                    std_png = std_dir / f"{tag}__{label}__{mode}__std_heatmap.png"

                    plot_std_heatmap(
                        std_contrast=std_c,
                        title=std_title,
                        out_path=std_png,
                        cmap=args.std_cmap,
                        vmax=std_vmax,
                        dpi=args.dpi,
                    )
                    print(f"Saved: {std_png}")

                # ------------------------------------------
                # COMBINED heatmaps (optional): routing + grounding
                # ------------------------------------------
                if args.make_combined:
                    if not args.make_std:
                        raise RuntimeError("--make-combined requires --make-std (needs std_vmax).")

                    combined_dir = file_outdir / "combined"
                    combined_dir.mkdir(exist_ok=True)

                    # std_c already computed above because you have:
                    # if args.make_std or args.make_combined: std_c = compute_std_contrast(...)

                    for variant in args.combined_modes:  # ["routing","grounding"] by default
                        comb_title = (
                            f"{tag} | {label} | mode={mode} | "
                            f"Attn_per-head layerwise (mean+std, {variant})"
                        )
                        comb_png = combined_dir / f"{tag}__{label}__{mode}__mean_std__{variant}.png"

                        plot_combined_heatmap(
                            mean_contrast=mean_c,
                            std_contrast=std_c,
                            title=comb_title,
                            out_path=comb_png,
                            cmap=args.cmap,
                            mean_vmax=vmax,
                            std_vmax=std_vmax,
                            min_alpha=args.min_alpha,
                            dpi=args.dpi,
                            variant=variant,                 # routing or grounding
                            alpha_cmap=args.alpha_cmap,
                        )
                        print(f"Saved: {comb_png}")



    print("Done.")



if __name__ == "__main__":
    main()






'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate comparable contrast heatmaps (video_mass - user_text_mass) from attention-per-head JSONL files.

For each input file:
- Output subfolder name extracted from filename stem:
  take the token immediately after 'attn_weights_per_head_200_16F_assistant_content_' and before next '_'
  e.g. '...content_spatial_video_mask...' -> 'spatial'
- Produce 6 heatmaps:
  - caption: none / vision / user_text
  - foil:    none / vision / user_text

Heatmap:
- background: mean contrast across items (layer x head)
- colormap: red ↔ blue (default RdBu_r), centered at 0, with shared global scale:
  vmax = pXX percentile of |contrast| across ALL input files (default p99)
- overlay markers (fixed size):
  - 'o' = top-3 video-oriented heads per layer (highest mean contrast)
  - 'x' = top-3 user_text-oriented heads per layer (lowest mean contrast)

Usage:
only with mean contrast heatmaps:
python make_attn_contrast_heatmaps.py \
  --input /path/to/attn/*.jsonl \
  --outdir /path/to/output_plots \
  --percentile 99 \
  --dpi 250

Optional (also generate STD heatmaps):
python make_attn_contrast_heatmaps.py \
  --input /path/to/attn/*.jsonl \
  --outdir /path/to/output_plots \
  --percentile 99 \
  --make-std \
  --std-cmap viridis \
  --dpi 250

  Optional (also generate combined mean+std heatmaps):
    In this heatmap, color encodes the direction of the modality bias 
    (video vs text), while opacity reflects the degree 
    of input-dependent variability.

  python attn_per_head_plots.py \
  --input file1.jsonl file2.jsonl \
  --outdir /path/to/plots \
  --percentile 99 \
  --make-std \
  --make-combined \
  --combined-modes routing grounding \
  --min-alpha 0.15 \
  --dpi 250

  here we can also decide to choose which combined variants to generate (also only one):
    --combined-modes routing grounding
  where:
    - routing   (Case A): alpha = 1 - std_norm  (stable => more opaque)
    - grounding (Case B): alpha = std_norm      (variable => more opaque    

"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


DEFAULT_PREFIX = "attn_weights_per_head_200_16F_assistant_content_"
DEFAULT_MODES = ["none", "vision", "user_text"]
DEFAULT_LABELS = ["caption", "foil"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot comparable contrast heatmaps (video - user_text) with global pXX scale across multiple JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", type=Path, nargs="+", required=True, help="One or more input .jsonl files")
    p.add_argument("--outdir", "-o", type=Path, required=True, help="Output directory (subfolders per file)")
    p.add_argument("--percentile", type=float, default=99.0, help="Global percentile for |contrast| to set vmax")
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="Mask modes to plot")
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, help="Labels to plot: caption and/or foil")
    p.add_argument("--cmap", type=str, default="RdBu_r", help="Matplotlib colormap (red↔blue recommended: RdBu_r)")
    p.add_argument("--dpi", type=int, default=200, help="DPI for saved images")
    p.add_argument("--marker-size", type=float, default=25.0, help="Fixed marker size (points^2) for X and O")
    p.add_argument("--linewidth", type=float, default=1.3, help="Marker linewidth")
    p.add_argument("--make-std", action="store_true", help="Also generate STD heatmaps")
    p.add_argument("--std-cmap", type=str, default="viridis", help="Colormap for STD heatmaps (sequential)")
    p.add_argument("--make-combined", action="store_true", help="Also generate combined mean+std heatmaps")
    
    p.add_argument("--min-alpha", type=float, default=0.15, help="Minimum alpha for combined heatmaps")
    

    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No JSONL records found in {path}")
    return rows


def sorted_layer_keys(results_for_label: Dict) -> List[str]:
    # expects keys like "layer_0" ... "layer_27"
    def layer_num(k: str) -> int:
        try:
            return int(k.split("_")[1])
        except Exception:
            return 10**9

    keys = list(results_for_label.keys())
    keys.sort(key=layer_num)
    return keys


def infer_shape_and_layers(rows: List[Dict], labels: List[str], modes: List[str]) -> Tuple[List[str], int]:
    """
    Find layer_keys and number of heads H from the first record that contains them.
    """
    for r in rows:
        results = r.get("results", {})
        for label in labels:
            if label in results and isinstance(results[label], dict) and results[label]:
                layer_keys = sorted_layer_keys(results[label])
                for mode in modes:
                    try:
                        payload = results[label][layer_keys[0]][mode]
                        H = len(payload["contrast_video_vs_user_text"])
                        return layer_keys, H
                    except Exception:
                        continue
    raise RuntimeError("Could not infer layer keys / head count from input JSONL.")


def compute_mean_contrast(rows: List[Dict], label: str, mode: str, layer_keys: List[str], H: int) -> np.ndarray:
    """
    Returns mean contrast matrix shape (L, H).
    """
    N = len(rows)
    L = len(layer_keys)
    all_c = np.zeros((N, L, H), dtype=np.float32)

    for i, r in enumerate(rows):
        res_label = r.get("results", {}).get(label, {})
        for l, lk in enumerate(layer_keys):
            try:
                c = res_label[lk][mode]["contrast_video_vs_user_text"]
                all_c[i, l, :] = np.asarray(c, dtype=np.float32)
            except Exception:
                # if missing, it stays zeros for that item/layer
                pass

    return all_c.mean(axis=0)  # (L, H)

def compute_std_contrast(rows: List[Dict], label: str, mode: str, layer_keys: List[str], H: int) -> np.ndarray:
    """
    Returns std contrast matrix shape (L, H), computed across items.
    """
    N = len(rows)
    L = len(layer_keys)
    all_c = np.zeros((N, L, H), dtype=np.float32)

    for i, r in enumerate(rows):
        res_label = r.get("results", {}).get(label, {})
        for l, lk in enumerate(layer_keys):
            try:
                c = res_label[lk][mode]["contrast_video_vs_user_text"]
                all_c[i, l, :] = np.asarray(c, dtype=np.float32)
            except Exception:
                pass

    return all_c.std(axis=0)  # (L,H)



def top3_by_mean(mean_contrast: np.ndarray, kind: str) -> np.ndarray:
    """
    mean_contrast shape (L, H)
    kind: "video" -> 3 largest; "user_text" -> 3 smallest
    Returns (L, 3) head indices.
    """
    L, H = mean_contrast.shape
    out = np.zeros((L, 3), dtype=int)

    for l in range(L):
        vals = mean_contrast[l]
        if kind == "video":
            out[l] = np.argsort(vals)[-3:][::-1]
        else:
            out[l] = np.argsort(vals)[:3]
    return out


def extract_subfolder_name(stem: str, prefix: str = DEFAULT_PREFIX) -> str:
    """
    Extract the token immediately after prefix and before next underscore.
    If prefix not found, fallback to last underscore chunk.
    """
    if prefix in stem:
        rest = stem.split(prefix, 1)[1]
        return rest.split("_", 1)[0] if "_" in rest else rest
    parts = stem.split("_")
    return parts[-1] if parts else stem


def compute_global_vmax(all_inputs: List[Path], labels: List[str], modes: List[str], percentile: float) -> float:
    """
    Compute global vmax as pXX of |mean_contrast| across all files/labels/modes.
    """
    abs_vals: List[float] = []

    for path in all_inputs:
        rows = load_jsonl(path)
        layer_keys, H = infer_shape_and_layers(rows, labels, modes)

        for label in labels:
            for mode in modes:
                mean_c = compute_mean_contrast(rows, label, mode, layer_keys, H)  # (L,H)
                abs_vals.append(np.abs(mean_c).ravel())

    abs_vals = np.concatenate(abs_vals, axis=0)
    vmax = float(np.percentile(abs_vals, percentile))
    # avoid degenerate scale
    if vmax <= 0:
        vmax = float(np.max(abs_vals)) if abs_vals.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
    return vmax

def compute_global_std_vmax(all_inputs: List[Path], labels: List[str], modes: List[str], percentile: float) -> float:
    """
    Compute global vmax for std as pXX of std_contrast across all files/labels/modes.
    """
    vals: List[np.ndarray] = []

    for path in all_inputs:
        rows = load_jsonl(path)
        layer_keys, H = infer_shape_and_layers(rows, labels, modes)

        for label in labels:
            for mode in modes:
                std_c = compute_std_contrast(rows, label, mode, layer_keys, H)  # (L,H)
                vals.append(std_c.ravel())

    vals = np.concatenate(vals, axis=0)
    vmax = float(np.percentile(vals, percentile))
    if vmax <= 0:
        vmax = float(np.max(vals)) if vals.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
    return vmax


def plot_heatmap(
    mean_contrast: np.ndarray,
    top_video: np.ndarray,
    top_text: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str,
    vmax: float,
    marker_size: float,
    linewidth: float,
    dpi: int,
):
    """
    mean_contrast: (L,H)
    top_video/top_text: (L,3) head indices
    """
    L, H = mean_contrast.shape
    img = mean_contrast.T  # (H, L)

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Head index")
    ax.set_title(title)

    # Colorbar con valori numerici + etichette semantiche
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean contrast (video_mass − user_text_mass)")
    cbar.ax.tick_params(labelsize=9)

    cbar.ax.text(
        0.5, 1.02, "video-oriented",
        ha="center", va="bottom",
        transform=cbar.ax.transAxes, fontsize=9
    )
    cbar.ax.text(
        0.5, -0.06, "text-oriented",
        ha="center", va="top",
        transform=cbar.ax.transAxes, fontsize=9
    )

    # Overlay X e O (3 + 3 per layer)
    for l in range(L):
        ax.scatter(
            np.full(3, l),
            top_video[l],
            s=marker_size,
            facecolors="none",
            edgecolors="black",
            marker="o",
            linewidths=linewidth,
            zorder=3,
        )
        ax.scatter(
            np.full(3, l),
            top_text[l],
            s=marker_size,
            c="black",
            marker="x",
            linewidths=linewidth,
            zorder=3,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_std_heatmap(
    std_contrast: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str,
    vmax: float,
    dpi: int,
):
    
    #std_contrast: (L,H)
    
    L, H = std_contrast.shape
    img = std_contrast.T  # (H,L)

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Head index")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Std of contrast across items")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    

def plot_combined_heatmap_caseB(
    mean_contrast: np.ndarray,   # (L,H)
    std_contrast: np.ndarray,    # (L,H)
    title: str,
    out_path: Path,
    cmap: str,
    mean_vmax: float,
    std_vmax: float,
    min_alpha: float,
    dpi: int,
):
    """
    Combined heatmap (Case B: variability-highlight):
      - RGB encodes mean contrast with diverging cmap (centered at 0)
      - Alpha encodes normalized std (higher std => more opaque / visible)

    This highlights cells where the modality bias is input-dependent.
    """
    L, H = mean_contrast.shape

    # (H,L) for imshow
    mean_img = mean_contrast.T
    std_img = std_contrast.T

    # Colormap for mean -> RGBA
    norm_mean = plt.Normalize(vmin=-mean_vmax, vmax=mean_vmax)
    cm = plt.get_cmap(cmap)
    rgba = cm(norm_mean(mean_img))  # (H,L,4), alpha initially 1

    # Case B: alpha = normalized std (clip to [0,1])
    std_norm = np.clip(std_img / (std_vmax + 1e-12), 0.0, 1.0)
    alpha = np.clip(std_norm, min_alpha, 1.0)
    rgba[..., 3] = alpha

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(rgba, aspect="auto", origin="lower", interpolation="nearest")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Head index")
    ax.set_title(title)

    # Colorbar for MEAN (values) only
    mappable = plt.cm.ScalarMappable(norm=norm_mean, cmap=cm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    cbar.set_label("Mean contrast (video_mass − user_text_mass)")
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.text(0.5, 1.02, "video-oriented", ha="center", va="bottom",
                 transform=cbar.ax.transAxes, fontsize=9)
    cbar.ax.text(0.5, -0.06, "text-oriented", ha="center", va="top",
                 transform=cbar.ax.transAxes, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


   





def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Global scales (comparable across all input files)
    # --------------------------------------------------
    # Mean contrast (diverging colormap, centered at 0)
    vmax = compute_global_vmax(
        all_inputs=args.input,
        labels=args.labels,
        modes=args.modes,
        percentile=args.percentile,
    )
    print(
        f"[INFO] Global mean vmax = p{args.percentile:.1f}(|mean_contrast|) "
        f"across all inputs = {vmax:.6f}"
    )

    # Std contrast (sequential colormap, >= 0)
    std_vmax = None
    if args.make_std:
        std_vmax = compute_global_std_vmax(
            all_inputs=args.input,
            labels=args.labels,
            modes=args.modes,
            percentile=args.percentile,
        )
        print(
            f"[INFO] Global std vmax = p{args.percentile:.1f}(std_contrast) "
            f"across all inputs = {std_vmax:.6f}"
        )

    # --------------------------------------------------
    # Per-file processing
    # --------------------------------------------------
    for in_path in args.input:
        rows = load_jsonl(in_path)
        layer_keys, H = infer_shape_and_layers(rows, args.labels, args.modes)

        stem = in_path.stem
        tag = extract_subfolder_name(stem, prefix=DEFAULT_PREFIX)

        file_outdir = args.outdir / tag
        file_outdir.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # >>> BLOCCO RICHIESTO (SEMPRE PRESENTE) <<<
        # --------------------------------------------------
        mean_dir = file_outdir / "mean"
        std_dir  = file_outdir / "std"
        mean_dir.mkdir(exist_ok=True)
        std_dir.mkdir(exist_ok=True)
        # --------------------------------------------------

        for label in args.labels:   # e.g. ["caption", "foil"]
            for mode in args.modes: # e.g. ["none", "vision", "user_text"]

                # ------------------------------------------
                # MEAN heatmap (with X/O markers)
                # ------------------------------------------
                mean_c = compute_mean_contrast(
                    rows, label, mode, layer_keys, H
                )  # shape (L, H)

                top_video = top3_by_mean(mean_c, kind="video")       # (L, 3)
                top_text  = top3_by_mean(mean_c, kind="user_text")   # (L, 3)

                mean_title = (
                    f"{tag} | {label} | mode={mode} | "
                    f"Attn_per-head layerwise"
                )
                mean_png = mean_dir / f"{tag}__{label}__{mode}__heatmap.png"

                plot_heatmap(
                    mean_contrast=mean_c,
                    top_video=top_video,
                    top_text=top_text,
                    title=mean_title,
                    out_path=mean_png,
                    cmap=args.cmap,
                    vmax=vmax,
                    marker_size=args.marker_size,
                    linewidth=args.linewidth,
                    dpi=args.dpi,
                )
                print(f"Saved: {mean_png}")

                # ------------------------------------------
                # STD heatmap (optional, no markers)
                # + reuse std_c for combined
                # ------------------------------------------
                std_c = None
                if args.make_std or args.make_combined:
                    std_c = compute_std_contrast(
                        rows, label, mode, layer_keys, H
                    )  # shape (L, H)

                    #check
                    std_norm = np.clip(std_c / (std_vmax + 1e-12), 0.0, 1.0)
                    print(tag, label, mode,
                        "std_vmax=", std_vmax,
                        "std min/median/p90/p99/max=",
                        float(std_c.min()),
                        float(np.median(std_c)),
                        float(np.percentile(std_c, 90)),
                        float(np.percentile(std_c, 99)),
                        float(std_c.max()),
                        "std_norm mean=", float(std_norm.mean()),
                        "std_norm p50/p90/p99=",
                        float(np.percentile(std_norm, 50)),
                        float(np.percentile(std_norm, 90)),
                        float(np.percentile(std_norm, 99))
                    )


                if args.make_std:
                    std_title = (
                        f"{tag} | {label} | mode={mode} | "
                        f"Attn_per-head layerwise (STD)"
                    )
                    std_png = std_dir / f"{tag}__{label}__{mode}__std_heatmap.png"

                    plot_std_heatmap(
                        std_contrast=std_c,
                        title=std_title,
                        out_path=std_png,
                        cmap=args.std_cmap,
                        vmax=std_vmax,
                        dpi=args.dpi,
                    )
                    print(f"Saved: {std_png}")

                # -----------------------------
                # COMBINED heatmap (Case B)
                # -----------------------------
                if args.make_combined:
                    if not args.make_std:
                        raise RuntimeError("--make-combined requires --make-std (needs std_vmax).")

                    # std_c must exist (compute it if you didn't already)
                    std_c = compute_std_contrast(rows, label, mode, layer_keys, H)

                    combined_dir = file_outdir / "combined"
                    combined_dir.mkdir(exist_ok=True)

                    combined_title = f"{tag} | {label} | mode={mode} | Attn_per-head layerwise (mean+std)"
                    combined_png = combined_dir / f"{tag}__{label}__{mode}__mean_std_heatmap.png"

                    plot_combined_heatmap_caseB(
                        mean_contrast=mean_c,
                        std_contrast=std_c,
                        title=combined_title,
                        out_path=combined_png,
                        cmap=args.cmap,
                        mean_vmax=vmax,
                        std_vmax=std_vmax,
                        min_alpha=args.min_alpha,
                        dpi=args.dpi,
                    )

                    print(f"Saved: {combined_png}")




    print("Done.")



if __name__ == "__main__":
    main()





'''








