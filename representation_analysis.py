#!/usr/bin/env python3
"""
Representation Analysis: Cross-modal Minibatch CKA and PCA
Compares Whisper (audio) vs LLMs (text) on paired LibriSpeech data.

Implements minibatch CKA using the unbiased HSIC_1 estimator from:
  Nguyen, Raghu & Kornblith (2021) "Do Wide and Deep Networks Learn the Same Things?"
  https://arxiv.org/abs/2010.15327

Which itself builds on:
  Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
  https://arxiv.org/abs/1905.00414

Output structure:
  Data/    — cached embeddings (.pkl) and numeric results (.json, .csv)
  Plots/   — all figures (.png)

Usage:
    # Full run (all LibriSpeech splits, ~280k utterances)
    python representation_analysis.py

    # Quick test run
    python representation_analysis.py --splits test.clean --max_samples 500

    # Re-run analysis using cached embeddings (skip re-extraction)
    python representation_analysis.py --skip_extraction

Models:
    - openai/whisper-base                       (audio, 74M)
    - znhoughton/opt-babylm-125m-20eps-seed964  (text, 125M, OPT, BabyLM)
    - znhoughton/opt-babylm-1.3b-20eps-seed964  (text, 1.3B, OPT, BabyLM)
    - allenai/OLMo-2-1124-7B                    (text, 7B, OLMo-2, Dolma)
    - EleutherAI/pythia-6.9b                    (text, 6.9B, GPT-NeoX, The Pile)
"""

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperModel,
    WhisperProcessor,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = {
    "whisper-base": {
        "hf_id": "openai/whisper-base",
        "modality": "audio",
        "params": "74M",
        "arch": "Whisper encoder",
        "corpus": "680k hrs audio",
    },
    "babylm-125m": {
        "hf_id": "znhoughton/opt-babylm-125m-20eps-seed964",
        "modality": "text",
        "params": "125M",
        "arch": "OPT",
        "corpus": "BabyLM",
    },
    "babylm-1.3b": {
        "hf_id": "znhoughton/opt-babylm-1.3b-20eps-seed964",
        "modality": "text",
        "params": "1.3B",
        "arch": "OPT",
        "corpus": "BabyLM",
    },
    "olmo-7b": {
        "hf_id": "allenai/OLMo-2-1124-7B",
        "modality": "text",
        "params": "7B",
        "arch": "OLMo-2",
        "corpus": "Dolma",
    },
    "pythia-6.9b": {
        "hf_id": "EleutherAI/pythia-6.9b",
        "modality": "text",
        "params": "6.9B",
        "arch": "GPT-NeoX",
        "corpus": "The Pile",
    },
}

ALL_SPLITS = [
    "train.clean.100",
    "train.clean.360",
    "train.other.500",
    "validation.clean",
    "validation.other",
    "test.clean",
    "test.other",
]

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30
MAX_TEXT_TOKENS = 512
PCA_COMPONENTS = 50

MINIBATCH_SIZE = 2048
MINIBATCH_SEED = 42

N_STABILITY_SUBSETS = 10
STABILITY_SUBSET_FRAC = 0.8

MODEL_COLORS = {
    "whisper-base":  "#1565C0",
    "babylm-125m":   "#E65100",
    "babylm-1.3b":   "#F9A825",
    "olmo-7b":       "#2E7D32",
    "pythia-6.9b":   "#6A1B9A",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available — running on CPU (slow for 7B models)")
    return device


def make_dirs(root: Path):
    (root / "Data").mkdir(parents=True, exist_ok=True)
    (root / "Plots").mkdir(parents=True, exist_ok=True)


PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
}

def _apply_style():
    plt.rcParams.update(PLOT_STYLE)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_librispeech(splits: list, max_samples: int | None):
    print(f"\n{'='*60}")
    print(f"Loading LibriSpeech splits: {splits}")
    print(f"{'='*60}")

    parts = []
    for split in splits:
        config = "clean" if "clean" in split else "other"
        print(f"  Fetching {split}…")
        ds = load_dataset(
            "openslr/librispeech_asr",
            config,
            split=split,
            streaming=False,
            trust_remote_code=True,
        )
        parts.append(ds)

    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    if max_samples and max_samples < len(dataset):
        rng = random.Random(MINIBATCH_SEED)
        indices = sorted(rng.sample(range(len(dataset)), max_samples))
        dataset = dataset.select(indices)

    texts = [s["text"].strip() for s in dataset]
    print(f"Total samples: {len(texts):,}")
    return dataset, texts


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_whisper_embeddings(model_id: str, dataset, device: torch.device) -> np.ndarray:
    """Mean-pool Whisper encoder hidden states over time frames."""
    print(f"\nLoading Whisper: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()

    embeddings = []
    for sample in tqdm(dataset, desc="whisper-base"):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        if sr != SAMPLE_RATE:
            audio = audio[:: int(sr / SAMPLE_RATE)]
        audio = audio[: MAX_AUDIO_SECONDS * SAMPLE_RATE]

        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        features = inputs.input_features.to(device, dtype=torch.float16)
        with torch.no_grad():
            enc = model.encoder(features)
            emb = enc.last_hidden_state.mean(dim=1).squeeze(0).float().cpu()
        embeddings.append(emb.numpy())

    del model
    torch.cuda.empty_cache()
    return np.stack(embeddings)


def extract_lm_embeddings(
    model_id: str,
    texts: list,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Mean-pool last hidden layer over non-padding tokens."""
    print(f"\nLoading LM: {model_id}")
    load_kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    label = model_id.split("/")[-1]
    for i in tqdm(range(0, len(texts), batch_size), desc=label):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_TOKENS,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1].float()
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embeddings.append(pooled.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Minibatch CKA  (Nguyen, Raghu & Kornblith 2021)
# ---------------------------------------------------------------------------

def _hsic1_batch(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased HSIC_1 estimator for a single minibatch (Song et al. 2007 U-statistic).

    HSIC_1(K, L) = 1/(n(n-3)) * [
        tr(K~ L~)
        + (1^T K~ 1)(1^T L~ 1) / ((n-1)(n-2))
        - 2/(n-2) * 1^T K~ L~ 1
    ]
    where K~, L~ have diagonal entries set to zero.

    This estimator is unbiased and independent of batch size, making it valid
    for averaging across minibatches (Nguyen et al. 2021, eq. 2).
    """
    n = X.shape[0]
    assert n >= 4, f"Minibatch size must be >= 4 for HSIC_1; got {n}"

    K = X @ X.T
    L = Y @ Y.T
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)

    KL = K @ L
    ones = np.ones(n)

    term1 = np.trace(KL)
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * (ones @ KL @ ones)

    return float((term1 + term2 - term3) / (n * (n - 3)))


def minibatch_cka(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = MINIBATCH_SIZE,
    seed: int = MINIBATCH_SEED,
) -> float:
    """
    Minibatch linear CKA (Nguyen, Raghu & Kornblith 2021).

    CKA_mb = mean_i[HSIC_1(K_i, L_i)]
            / sqrt(mean_i[HSIC_1(K_i, K_i)] * mean_i[HSIC_1(L_i, L_i)])

    Samples are shuffled once and consumed in non-overlapping minibatches.
    The score is invariant to rotation and isotropic scaling of the embedding
    spaces, and is unbiased w.r.t. batch size.

    Note: per Murphy et al. (2024), this estimator has a slight downward bias
    relative to the true population CKA, but this affects all pairs equally
    so relative comparisons remain valid.
    """
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)

    hsic_xy, hsic_xx, hsic_yy = [], [], []

    for start in range(0, N - batch_size + 1, batch_size):
        idx = indices[start : start + batch_size]
        Xb = X[idx].astype(np.float64)
        Yb = Y[idx].astype(np.float64)
        # L2-normalise within batch for numerical stability
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-10
        Yb /= np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-10

        hsic_xy.append(_hsic1_batch(Xb, Yb))
        hsic_xx.append(_hsic1_batch(Xb, Xb))
        hsic_yy.append(_hsic1_batch(Yb, Yb))

    # Edge case: fewer samples than one batch
    if not hsic_xy:
        Xb = X.astype(np.float64)
        Yb = Y.astype(np.float64)
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-10
        Yb /= np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-10
        denom = np.sqrt(_hsic1_batch(Xb, Xb) * _hsic1_batch(Yb, Yb))
        return float(_hsic1_batch(Xb, Yb) / (denom + 1e-10))

    mean_xy = float(np.mean(hsic_xy))
    denom = np.sqrt(max(float(np.mean(hsic_xx)), 0.0) * max(float(np.mean(hsic_yy)), 0.0))
    return float(mean_xy / denom) if denom > 1e-10 else 0.0


def cka_stability_check(
    X: np.ndarray,
    Y: np.ndarray,
    n_subsets: int = N_STABILITY_SUBSETS,
    subset_frac: float = STABILITY_SUBSET_FRAC,
    batch_size: int = MINIBATCH_SIZE,
) -> dict:
    """
    Assess outlier sensitivity (Horoi et al.) by computing CKA on random subsets.
    High std across subsets flags potential outlier influence.
    """
    N = X.shape[0]
    k = int(N * subset_frac)
    scores = []
    rng = np.random.default_rng(MINIBATCH_SEED + 999)
    for i in range(n_subsets):
        idx = rng.choice(N, size=k, replace=False)
        scores.append(minibatch_cka(X[idx], Y[idx], batch_size=batch_size, seed=i))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": [float(s) for s in scores],
    }


# ---------------------------------------------------------------------------
# PCA / eigenvalue analysis
# ---------------------------------------------------------------------------

def pca_eigenvalues(X: np.ndarray, n_components: int = PCA_COMPONENTS) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    _, s, _ = np.linalg.svd(Xc / np.sqrt(n - 1), full_matrices=False)
    eigs = s ** 2
    eigs /= eigs.sum()
    return eigs[:n_components]


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalised eigenvalue distribution)."""
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_eigenspectra(eigenvalues: dict, plots_dir: Path):
    _apply_style()
    names = list(eigenvalues.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        eigs = eigenvalues[name]
        color = MODEL_COLORS.get(name, "#555555")
        k = np.arange(1, len(eigs) + 1)
        ax.fill_between(k, eigs, alpha=0.18, color=color)
        ax.plot(k, eigs, color=color, linewidth=2.2)
        er = effective_rank(eigs)
        ax.set_title(f"{name}\n(eff. rank = {er:.1f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Principal Component", fontsize=9)
        ax.set_ylabel("Fraction of Variance", fontsize=9)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    fig.suptitle(
        "PCA Eigenvalue Spectra  ·  similar decay → similar intrinsic dimensionality",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = plots_dir / "eigenspectra.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_eigenspectra_overlay(eigenvalues: dict, plots_dir: Path):
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, eigs in eigenvalues.items():
        color = MODEL_COLORS.get(name, "#555555")
        k = np.arange(1, len(eigs) + 1)
        ax.plot(k, eigs, color=color, linewidth=2, label=name, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Fraction of Variance (log)", fontsize=11)
    ax.set_title("Eigenvalue Spectra — All Models Overlaid", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    plt.tight_layout()
    path = plots_dir / "eigenspectra_overlay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_effective_rank_bar(eigenvalues: dict, plots_dir: Path):
    _apply_style()
    names = list(eigenvalues.keys())
    ranks = [effective_rank(eigenvalues[n]) for n in names]
    colors = [MODEL_COLORS.get(n, "#555") for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, ranks, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, rank in zip(bars, ranks):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{rank:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title(
        "Effective Rank of Representation Spaces\nHigher = variance spread across more dimensions",
        fontsize=11, fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()
    path = plots_dir / "effective_rank.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_pca_scatter(embeddings: dict, plots_dir: Path):
    from sklearn.decomposition import PCA as skPCA
    _apply_style()
    names = list(embeddings.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        X = embeddings[name]
        color = MODEL_COLORS.get(name, "#555")
        pca = skPCA(n_components=2)
        Z = pca.fit_transform(X)
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.35, s=12, color=color, linewidths=0)
        var = pca.explained_variance_ratio_
        ax.set_title(name, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=8)

    fig.suptitle("2D PCA Projections  ·  each point = one utterance", fontsize=12, y=1.01)
    plt.tight_layout()
    path = plots_dir / "pca_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_cka_heatmap(cka_matrix: np.ndarray, names: list, plots_dir: Path):
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6.5))

    cmap = LinearSegmentedColormap.from_list(
        "cka", ["#FFFFFF", "#A5D6A7", "#2E7D32"], N=256
    )
    im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA  (0 = no similarity,  1 = identical)", fontsize=9)

    n = len(names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)

    for i in range(n):
        for j in range(n):
            val = cka_matrix[i, j]
            text_color = "black" if val < 0.7 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    audio_indices = [i for i, name in enumerate(names) if MODELS[name]["modality"] == "audio"]
    if audio_indices:
        boundary = max(audio_indices) + 0.5
        ax.axhline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.axvline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_title(
        "Pairwise Minibatch Linear CKA\n"
        "Dashed line separates audio (Whisper) from text models",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    path = plots_dir / "cka_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_cka_bar_cross_modal(cka_matrix: np.ndarray, names: list, plots_dir: Path):
    """Bar chart: each LLM's CKA with Whisper."""
    _apply_style()
    if "whisper-base" not in names:
        return
    w_idx = names.index("whisper-base")
    lm_names = [n for n in names if n != "whisper-base"]
    scores = [cka_matrix[w_idx, names.index(n)] for n in lm_names]
    colors = [MODEL_COLORS.get(n, "#555") for n in lm_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(lm_names, scores, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylim(0, min(1.0, max(scores) * 1.35 + 0.05))
    ax.set_ylabel("CKA with Whisper", fontsize=11)
    ax.set_title(
        "Cross-Modal CKA: Each LLM vs. Whisper\n"
        "Higher = more similar representational geometry to audio encoder",
        fontsize=11, fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()
    path = plots_dir / "cka_cross_modal_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_stability(stability_results: dict, plots_dir: Path):
    """Box plot of CKA variance across random subsets per pair."""
    _apply_style()
    pairs = list(stability_results.keys())
    data = [stability_results[p]["scores"] for p in pairs]
    means = [stability_results[p]["mean"] for p in pairs]
    stds = [stability_results[p]["std"] for p in pairs]

    order = np.argsort(means)[::-1]
    pairs = [pairs[i] for i in order]
    data = [data[i] for i in order]
    stds_sorted = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.5), 5))
    bp = ax.boxplot(
        data, patch_artist=True, vert=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, std in zip(bp["boxes"], stds_sorted):
        intensity = min(std / 0.05, 1.0)
        patch.set_facecolor((0.2 + 0.6 * intensity, 0.6 - 0.4 * intensity, 0.8 - 0.6 * intensity, 0.5))

    ax.set_xticks(range(1, len(pairs) + 1))
    ax.set_xticklabels(
        [p.replace(" vs ", "\nvs\n") for p in pairs],
        fontsize=8,
    )
    ax.set_ylabel("CKA Score", fontsize=11)
    ax.set_title(
        f"CKA Stability: {N_STABILITY_SUBSETS} random {int(STABILITY_SUBSET_FRAC*100)}% subsets per pair\n"
        "Red tint = high variance (potential outlier sensitivity per Horoi et al.)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path = plots_dir / "cka_stability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Summary tables (CSV)
# ---------------------------------------------------------------------------

def save_summary_tables(
    cka_matrix: np.ndarray,
    names: list,
    eigenvalues: dict,
    embeddings: dict,
    stability_results: dict,
    data_dir: Path,
):
    # Table 1: full pairwise CKA matrix
    path1 = data_dir / "cka_matrix.csv"
    with open(path1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, row_name in enumerate(names):
            w.writerow([row_name] + [f"{cka_matrix[i, j]:.6f}" for j in range(len(names))])
    print(f"  Saved → {path1}")

    # Table 2: cross-modal (Whisper vs LLMs)
    if "whisper-base" in names:
        w_idx = names.index("whisper-base")
        path2 = data_dir / "cka_cross_modal.csv"
        with open(path2, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "params", "arch", "corpus", "cka_with_whisper"])
            for name in names:
                if name == "whisper-base":
                    continue
                idx = names.index(name)
                w.writerow([
                    name,
                    MODELS[name]["params"],
                    MODELS[name]["arch"],
                    MODELS[name]["corpus"],
                    f"{cka_matrix[w_idx, idx]:.6f}",
                ])
        print(f"  Saved → {path2}")

    # Table 3: PCA / effective rank
    path3 = data_dir / "pca_summary.csv"
    with open(path3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "params", "arch", "corpus",
            "embedding_dim", "effective_rank",
            "var_pc1", "var_top5", "var_top10",
        ])
        for name, eigs in eigenvalues.items():
            dim = embeddings[name].shape[1] if name in embeddings else "—"
            w.writerow([
                name,
                MODELS[name]["params"],
                MODELS[name]["arch"],
                MODELS[name]["corpus"],
                dim,
                f"{effective_rank(eigs):.2f}",
                f"{eigs[0]:.4f}",
                f"{eigs[:5].sum():.4f}",
                f"{eigs[:10].sum():.4f}",
            ])
    print(f"  Saved → {path3}")

    # Table 4: stability
    if stability_results:
        path4 = data_dir / "cka_stability.csv"
        with open(path4, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "cka_mean", "cka_std", "cka_min", "cka_max", "unstable"])
            for pair, stats in stability_results.items():
                scores = stats["scores"]
                w.writerow([
                    pair,
                    f"{stats['mean']:.6f}",
                    f"{stats['std']:.6f}",
                    f"{min(scores):.6f}",
                    f"{max(scores):.6f}",
                    "yes" if stats["std"] > 0.03 else "no",
                ])
        print(f"  Saved → {path4}")


def print_summary(cka_matrix: np.ndarray, names: list, stability_results: dict):
    print(f"\n{'='*70}")
    print("PAIRWISE CKA  (upper triangle)")
    print(f"{'='*70}")
    print(f"  {'Pair':<42} {'CKA':>7}  {'±Std':>7}  Note")
    print("  " + "-" * 66)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            score = cka_matrix[i, j]
            pair_key = f"{a} vs {b}"
            std = stability_results.get(pair_key, {}).get("std", 0.0)
            flag = "  ⚠ unstable" if std > 0.03 else ""
            print(f"  {pair_key:<42} {score:>7.4f}  {std:>7.4f}{flag}")

    print(f"\n{'='*70}")
    print("CROSS-MODAL CKA  (Whisper ↔ LLMs)")
    print(f"{'='*70}")
    if "whisper-base" in names:
        w_idx = names.index("whisper-base")
        for name in names:
            if name == "whisper-base":
                continue
            score = cka_matrix[w_idx, names.index(name)]
            meta = MODELS[name]
            print(
                f"  whisper-base ↔ {name:<18} "
                f"[{meta['params']:>5} | {meta['arch']:<12} | {meta['corpus']}]"
                f"  CKA = {score:.4f}"
            )

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("  CKA ~1.0  identical geometry")
    print("  CKA ~0.5  moderate similarity")
    print("  CKA ~0.0  no shared structure")
    print()
    print("  Cross-modal CKA (Whisper ↔ LLM) > ~0.3 supports the Platonic")
    print("  Representation Hypothesis (convergence across modalities).")
    print("  ⚠  std > 0.03 across subsets → possible outlier sensitivity")
    print("     (see Horoi et al. 'Deceiving the CKA Similarity Measure')")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-modal representation analysis with minibatch CKA"
    )
    p.add_argument(
        "--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS,
        help="LibriSpeech splits to use (default: all)",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap total samples after concatenation (default: no cap)",
    )
    p.add_argument(
        "--root_dir", type=str, default=".",
        help="Root directory; Data/ and Plots/ subdirs created here (default: .)",
    )
    p.add_argument(
        "--skip_extraction", action="store_true",
        help="Load cached embeddings from Data/ instead of re-extracting",
    )
    p.add_argument(
        "--batch_size", type=int, default=MINIBATCH_SIZE,
        help=f"Minibatch size for CKA (default: {MINIBATCH_SIZE})",
    )
    p.add_argument(
        "--lm_batch_size", type=int, default=8,
        help="Batch size for LM forward passes (default: 8)",
    )
    p.add_argument(
        "--pca_components", type=int, default=PCA_COMPONENTS,
        help=f"PCA components for eigenvalue analysis (default: {PCA_COMPONENTS})",
    )
    p.add_argument(
        "--skip_stability", action="store_true",
        help="Skip outlier stability check (faster)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root = Path(args.root_dir)
    data_dir = root / "Data"
    plots_dir = root / "Plots"
    make_dirs(root)
    device = get_device()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset, texts = load_librispeech(args.splits, args.max_samples)
    N = len(texts)

    # ------------------------------------------------------------------
    # 2. Extract / load embeddings (one model at a time)
    # ------------------------------------------------------------------
    embeddings = {}
    for model_name, cfg in MODELS.items():
        cache_path = data_dir / f"embeddings_{model_name}.pkl"

        if args.skip_extraction and cache_path.exists():
            print(f"\nLoading cached: {model_name}")
            with open(cache_path, "rb") as f:
                embeddings[model_name] = pickle.load(f)
            print(f"  Shape: {embeddings[model_name].shape}")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting: {model_name}  [{cfg['params']} | {cfg['arch']}]")
        print(f"{'='*60}")

        if cfg["modality"] == "audio":
            emb = extract_whisper_embeddings(cfg["hf_id"], dataset, device)
        else:
            emb = extract_lm_embeddings(cfg["hf_id"], texts, device, batch_size=args.lm_batch_size)

        print(f"  Shape: {emb.shape}")
        embeddings[model_name] = emb
        with open(cache_path, "wb") as f:
            pickle.dump(emb, f)
        print(f"  Cached → {cache_path}")

    names = list(embeddings.keys())

    # ------------------------------------------------------------------
    # 3. PCA eigenvalue analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PCA eigenvalue analysis…")
    print(f"{'='*60}")

    eigenvalues = {}
    for name, X in embeddings.items():
        eigenvalues[name] = pca_eigenvalues(X, n_components=args.pca_components)
        er = effective_rank(eigenvalues[name])
        print(f"  {name:<22}  dim={X.shape[1]:>5}  eff_rank={er:.1f}")

    print("\nGenerating PCA plots…")
    plot_eigenspectra(eigenvalues, plots_dir)
    plot_eigenspectra_overlay(eigenvalues, plots_dir)
    plot_effective_rank_bar(eigenvalues, plots_dir)
    try:
        from sklearn.decomposition import PCA  # noqa: F401
        plot_pca_scatter(embeddings, plots_dir)
    except ImportError:
        print("scikit-learn not installed — skipping PCA scatter")

    # ------------------------------------------------------------------
    # 4. Pairwise minibatch CKA
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Pairwise minibatch CKA  (batch_size={args.batch_size}, N={N:,})…")
    print(f"{'='*60}")

    n = len(names)
    cka_matrix = np.zeros((n, n))
    cka_results = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                cka_matrix[i, j] = 1.0
                continue
            if j < i:
                cka_matrix[i, j] = cka_matrix[j, i]
                continue
            score = minibatch_cka(
                embeddings[names[i]], embeddings[names[j]],
                batch_size=args.batch_size,
            )
            cka_matrix[i, j] = score
            key = f"{names[i]} vs {names[j]}"
            cka_results[key] = score
            print(f"  CKA({names[i]}, {names[j]}) = {score:.4f}")

    print("\nGenerating CKA plots…")
    plot_cka_heatmap(cka_matrix, names, plots_dir)
    plot_cka_bar_cross_modal(cka_matrix, names, plots_dir)

    # ------------------------------------------------------------------
    # 5. Outlier stability check  (Horoi et al.)
    # ------------------------------------------------------------------
    stability_results = {}
    if not args.skip_stability:
        print(f"\n{'='*60}")
        print(
            f"Outlier stability check  "
            f"({N_STABILITY_SUBSETS} subsets × {int(STABILITY_SUBSET_FRAC*100)}% of data)…"
        )
        print(f"{'='*60}")
        for i in range(n):
            for j in range(n):
                if j <= i:
                    continue
                pair_key = f"{names[i]} vs {names[j]}"
                stats = cka_stability_check(
                    embeddings[names[i]], embeddings[names[j]],
                    batch_size=args.batch_size,
                )
                stability_results[pair_key] = stats
                flag = "  ⚠" if stats["std"] > 0.03 else ""
                print(
                    f"  {pair_key:<42}  "
                    f"mean={stats['mean']:.4f}  std={stats['std']:.4f}{flag}"
                )
        plot_stability(stability_results, plots_dir)
    else:
        print("\nStability check skipped (--skip_stability)")

    # ------------------------------------------------------------------
    # 6. Save all outputs
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Saving summary tables…")
    print(f"{'='*60}")

    save_summary_tables(cka_matrix, names, eigenvalues, embeddings, stability_results, data_dir)

    results_json = {
        "n_samples": N,
        "splits_used": args.splits,
        "minibatch_size": args.batch_size,
        "models": {k: v for k, v in MODELS.items()},
        "embedding_shapes": {k: list(v.shape) for k, v in embeddings.items()},
        "cka_scores": cka_results,
        "cka_matrix": cka_matrix.tolist(),
        "eigenvalue_spectra": {k: v.tolist() for k, v in eigenvalues.items()},
        "effective_ranks": {k: effective_rank(v) for k, v in eigenvalues.items()},
        "stability": {
            k: {"mean": v["mean"], "std": v["std"]}
            for k, v in stability_results.items()
        },
    }
    json_path = data_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved → {json_path}")

    # ------------------------------------------------------------------
    # 7. Console summary
    # ------------------------------------------------------------------
    print_summary(cka_matrix, names, stability_results)

    print(f"\nOutputs:")
    print(f"  Data/   {data_dir.resolve()}")
    print(f"  Plots/  {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
