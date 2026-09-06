#!/usr/bin/env python3
"""
Standalone script to generate off-diagonal cosine-similarity histograms
from cached word embedding pickles produced by the word-level analysis scripts.

Scans MLSData/ and MCVData/ for cached embedding files and writes one
histogram per model into the corresponding plots folder:
  MLSPlots/similarity_histograms/
  MCVPlots/similarity_histograms/

Usage
-----
  python generate_similarity_histograms.py
  python generate_similarity_histograms.py --max_n 5000  # larger subsample
  python generate_similarity_histograms.py --data_dirs MLSData  # one dataset only
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANALYSES = [
    {"data_dir": "WordData", "plots_dir": "WordPlots", "prefix": "word_"},
    {"data_dir": "MLSData",  "plots_dir": "MLSPlots",  "prefix": "mls_"},
    {"data_dir": "MCVData",  "plots_dir": "MCVPlots",  "prefix": "mcv_",
     "word_records": "mcv_word_records.json"},
]


def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _off_diagonal_sims(embeddings, max_n=3000, seed=42, word_ids=None, speaker_ids=None, normalize=False):
    sims = {}
    for model_name, emb in embeddings.items():
        N = emb.shape[0]
        X = emb.astype(np.float32)

        if word_ids is not None and speaker_ids is not None:
            # Group by (word, sentence, position) key and compute all
            # cross-speaker pairs within each group — no subsampling needed.
            if normalize:
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X = X / np.maximum(norms, 1e-10)
            else:
                X = X / X.shape[1]

            all_vals = []
            for wid in np.unique(word_ids):
                group_idx = np.where(word_ids == wid)[0]
                if len(group_idx) < 2:
                    continue
                Xg  = X[group_idx]
                sids_g = speaker_ids[group_idx]
                sim_g = Xg @ Xg.T
                n_g = len(group_idx)
                cross_spk = sids_g[:, None] != sids_g[None, :]
                off_diag  = ~np.eye(n_g, dtype=bool)
                mask = cross_spk & off_diag
                all_vals.append(sim_g[mask])

            vals = np.concatenate(all_vals) if all_vals else np.array([])
            note = f"n={N:,} (all cross-speaker pairs)"
        else:
            # Fallback: random subsample + full off-diagonal
            rng = np.random.default_rng(seed)
            if N > max_n:
                idx = rng.choice(N, max_n, replace=False)
                X = X[idx]
                N = max_n
            if normalize:
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X = X / np.maximum(norms, 1e-10)
            sim = X @ X.T if normalize else (X @ X.T) / X.shape[1]
            vals = sim[~np.eye(N, dtype=bool)]
            note = f"n={N:,}"

        sims[model_name] = (vals, note)
    return sims


def _plot_sim_suite(sims, hist_dir, prefix, file_tag, xlabel, overlay_xlim, color_map):
    names = list(sims.keys())
    n_models = len(names)

    for model_name, (off_diag, note) in sims.items():
        mean_val = float(off_diag.mean())
        plot_vals = np.clip(off_diag, overlay_xlim[0], overlay_xlim[1])
        _, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(plot_vals, bins=100, range=overlay_xlim, color=color_map[model_name], edgecolor="none", alpha=0.85)
        ax.axvline(mean_val, color="#E53935", linewidth=1.5, linestyle="--",
                   label=f"mean = {mean_val:.3f}")
        ax.set_xlim(*overlay_xlim)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{model_name}  —  off-diagonal {xlabel.lower()}  ({note})", fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = hist_dir / f"{prefix}{model_name}_{file_tag}_hist.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")

    _, ax = plt.subplots(figsize=(10, 5))
    for model_name, (off_diag, _) in sims.items():
        ax.hist(off_diag, bins=80, histtype="step", linewidth=1.5, density=True,
                color=color_map[model_name], label=model_name)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Off-diagonal {xlabel.lower()} — all models", fontsize=11, fontweight="bold")
    ax.set_xlim(*overlay_xlim)
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    path = hist_dir / f"{prefix}all_models_{file_tag}_overlay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8), squeeze=False)
    axes_flat = axes.flatten()
    for idx, model_name in enumerate(names):
        off_diag, _ = sims[model_name]
        ax = axes_flat[idx]
        ax.hist(np.clip(off_diag, overlay_xlim[0], overlay_xlim[1]), bins=60, range=overlay_xlim,
                color=color_map[model_name], edgecolor="none", alpha=0.85)
        ax.set_xlim(*overlay_xlim)
        ax.set_title(model_name, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle(f"Off-diagonal {xlabel.lower()} — per model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = hist_dir / f"{prefix}all_models_{file_tag}_faceted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    means = [float(sims[name][0].mean()) for name in names]
    _, ax = plt.subplots(figsize=(max(8, n_models * 0.7), 4.5))
    ax.bar(names, means, color=[color_map[n] for n in names], edgecolor="none", alpha=0.85)
    ax.set_ylabel(f"Mean {xlabel}", fontsize=11)
    ax.set_title(f"Mean off-diagonal {xlabel.lower()} per model", fontsize=11, fontweight="bold")
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    path = hist_dir / f"{prefix}all_models_{file_tag}_mean_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    sds = [float(sims[name][0].std()) for name in names]
    _, ax = plt.subplots(figsize=(max(8, n_models * 0.7), 4.5))
    ax.bar(names, sds, color=[color_map[n] for n in names], edgecolor="none", alpha=0.85)
    ax.set_ylabel(f"SD of {xlabel}", fontsize=11)
    ax.set_title(f"SD of off-diagonal {xlabel.lower()} per model", fontsize=11, fontweight="bold")
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    path = hist_dir / f"{prefix}all_models_{file_tag}_sd_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_similarity_histograms(embeddings, plots_dir, prefix, max_n=3000, seed=42, word_ids=None, speaker_ids=None):
    _apply_style()
    hist_dir = plots_dir / "similarity_histograms"
    hist_dir.mkdir(exist_ok=True)

    names = list(embeddings.keys())
    n_models = len(names)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_models, 1)))
    color_map = dict(zip(names, colors))

    _plot_sim_suite(
        _off_diagonal_sims(embeddings, max_n=max_n, seed=seed, word_ids=word_ids, speaker_ids=speaker_ids),
        hist_dir, prefix, "dot_product_sim", "Dot Product Similarity", (-10, 10), color_map,
    )
    _plot_sim_suite(
        _off_diagonal_sims(embeddings, max_n=max_n, seed=seed, word_ids=word_ids, speaker_ids=speaker_ids, normalize=True),
        hist_dir, prefix, "cosine_sim", "Cosine Similarity", (-1, 1), color_map,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate cosine-similarity histograms.")
    parser.add_argument("--max_n", type=int, default=3000,
                        help="Max words to subsample for similarity matrix (default: 3000)")
    parser.add_argument("--data_dirs", nargs="+",
                        help="Restrict to specific data dirs (e.g. MLSData MCVData)")
    args = parser.parse_args()

    root = Path(__file__).parent

    for analysis in ANALYSES:
        data_dir  = root / analysis["data_dir"]
        plots_dir = root / analysis["plots_dir"]
        prefix    = analysis["prefix"]

        if args.data_dirs and analysis["data_dir"] not in args.data_dirs:
            continue
        if not data_dir.exists():
            print(f"Skipping {data_dir} (not found)")
            continue

        pkl_files = sorted(data_dir.glob(f"{prefix}word_embeddings_*.pkl"))
        if not pkl_files:
            print(f"No embedding pickles found in {data_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {analysis['data_dir']}  ({len(pkl_files)} models)")
        print(f"{'='*60}")

        embeddings = {}
        for pkl in pkl_files:
            model_name = re.sub(rf"^{re.escape(prefix)}word_embeddings_", "", pkl.stem)
            print(f"  Loading {model_name} ...", end=" ", flush=True)
            with open(pkl, "rb") as f:
                emb = pickle.load(f)
            print(f"shape={emb.shape}")
            embeddings[model_name] = emb

        # Apply valid_mask (same logic as in the analysis scripts)
        N = next(iter(embeddings.values())).shape[0]
        valid_mask = np.ones(N, dtype=bool)
        for emb in embeddings.values():
            valid_mask &= np.linalg.norm(emb, axis=1) > 1e-10
        for name in embeddings:
            embeddings[name] = embeddings[name][valid_mask]

        # Build word_ids and speaker_ids for same-word/cross-speaker masking
        word_ids    = None
        speaker_ids = None
        word_records_file = analysis.get("word_records")
        if word_records_file:
            wr_path = data_dir / word_records_file
            if wr_path.exists():
                with open(wr_path) as f:
                    word_records = json.load(f)

                # word_ids: (word, sentence, char_start) — matches build_word_type_ids
                all_types: dict = {}
                ids = []
                for rec in word_records:
                    full_key = (rec["word"], rec["sentence"], rec["char_start"])
                    if full_key not in all_types:
                        all_types[full_key] = len(all_types)
                    ids.append(all_types[full_key])
                word_ids = np.array(ids, dtype=np.int32)[valid_mask]

                # speaker_ids: integer per unique speaker
                all_speakers = sorted(set(rec.get("speaker_id", "unknown") for rec in word_records))
                spk_to_id = {s: i for i, s in enumerate(all_speakers)}
                speaker_ids = np.array(
                    [spk_to_id[rec.get("speaker_id", "unknown")] for rec in word_records],
                    dtype=np.int32
                )[valid_mask]

                print(f"  Word types (word+sentence+pos): {len(all_types):,}, speakers: {len(all_speakers):,}")
            else:
                print(f"  Warning: {wr_path} not found, skipping word/speaker mask")

        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_similarity_histograms(embeddings, plots_dir, prefix=prefix, max_n=args.max_n,
                                   word_ids=word_ids, speaker_ids=speaker_ids)

    print("\nDone.")


if __name__ == "__main__":
    main()
