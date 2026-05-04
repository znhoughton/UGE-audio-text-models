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
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANALYSES = [
    {"data_dir": "WordData", "plots_dir": "WordPlots", "prefix": "word_"},
    {"data_dir": "MLSData",  "plots_dir": "MLSPlots",  "prefix": "mls_"},
    {"data_dir": "MCVData",  "plots_dir": "MCVPlots",  "prefix": "mcv_"},
]


def plot_similarity_histograms(embeddings, plots_dir, prefix, max_n=3000, seed=42):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    hist_dir = plots_dir / "similarity_histograms"
    hist_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    for model_name, emb in embeddings.items():
        N = emb.shape[0]
        X = emb.astype(np.float32)
        if N > max_n:
            idx = rng.choice(N, max_n, replace=False)
            X = X[idx]

        sim = X @ X.T
        n = sim.shape[0]
        off_diag = sim[~np.eye(n, dtype=bool)]

        mean_val = float(off_diag.mean())
        sample_note = f"subsample n={n:,}/{N:,}" if N > max_n else f"n={n:,}"

        _, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(off_diag, bins=100, color="#4A90D9", edgecolor="none", alpha=0.85)
        ax.axvline(mean_val, color="#E53935", linewidth=1.5, linestyle="--",
                   label=f"mean = {mean_val:.3f}")
        ax.set_xlabel("Dot Product Similarity", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{model_name}  —  off-diagonal cosine similarity  ({sample_note})",
                     fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = hist_dir / f"{prefix}{model_name}_cosine_sim_hist.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")


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

        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_similarity_histograms(embeddings, plots_dir, prefix=prefix, max_n=args.max_n)

    print("\nDone.")


if __name__ == "__main__":
    main()
