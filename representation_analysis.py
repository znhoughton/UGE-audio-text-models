#!/usr/bin/env python3
"""
Representation Analysis: Cross-modal CKA and PCA
Compares Whisper (audio) vs LLMs (text) on paired LibriSpeech data.

Usage:
    python representation_analysis.py [--n_samples N] [--output_dir DIR] [--skip_existing]

Models compared:
    - openai/whisper-base                       (audio encoder)
    - znhoughton/opt-babylm-125m-20eps-seed964  (text, 125M, OPT-based)
    - znhoughton/opt-babylm-1.3b-20eps-seed964  (text, 1.3B, OPT-based)
    - allenai/OLMo-2-1124-7B                    (text, 7B, OLMo-2 arch)
    - EleutherAI/pythia-6.9b                    (text, 6.9B, GPT-NeoX arch — parallel attn+MLP, The Pile)
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datasets import load_dataset
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
    },
    "babylm-125m": {
        "hf_id": "znhoughton/opt-babylm-125m-20eps-seed964",
        "modality": "text",
    },
    "babylm-1.3b": {
        "hf_id": "znhoughton/opt-babylm-1.3b-20eps-seed964",
        "modality": "text",
    },
    "olmo-7b": {
        "hf_id": "allenai/OLMo-2-1124-7B",
        "modality": "text",
    },
    "pythia-6.9b": {
        "hf_id": "EleutherAI/pythia-6.9b",
        "modality": "text",
    },
}

DATASET_NAME = "openslr/librispeech_asr"
DATASET_SPLIT = "test.clean"
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30  # Whisper's context window
MAX_TEXT_TOKENS = 512
PCA_COMPONENTS = 50     # for eigenvalue spectrum analysis


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available, falling back to CPU (will be slow)")
    return device


def extract_whisper_embeddings(model_id: str, samples: list, device: torch.device) -> np.ndarray:
    """Mean-pool Whisper encoder hidden states over time frames."""
    print(f"\nLoading Whisper: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()

    embeddings = []
    for sample in tqdm(samples, desc="Whisper audio"):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]

        # Resample if needed (simple truncation to SAMPLE_RATE assumption;
        # for production use torchaudio.transforms.Resample)
        if sr != SAMPLE_RATE:
            # crude decimation — good enough for same-model comparisons
            ratio = sr / SAMPLE_RATE
            audio = audio[:: int(ratio)]

        # Clip to Whisper's 30s context
        max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
        audio = audio[:max_samples]

        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device, dtype=torch.float16)

        with torch.no_grad():
            encoder_out = model.encoder(input_features)
            # Shape: (1, T, hidden_dim) → mean pool over T
            emb = encoder_out.last_hidden_state.mean(dim=1).squeeze(0)

        embeddings.append(emb.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.stack(embeddings)  # (N, hidden_dim)


def extract_lm_embeddings(model_id: str, texts: list[str], device: torch.device) -> np.ndarray:
    """Mean-pool last hidden states over non-padding tokens for a causal LM."""
    print(f"\nLoading LM: {model_id}")

    # OLMo uses AutoModelForCausalLM; OPT-based BabyLM also works via AutoModel
    # We try CausalLM first (gives us hidden states), fall back to AutoModel
    load_kwargs = dict(
        torch_dtype=torch.float16,
        output_hidden_states=False,
        trust_remote_code=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)

    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embeddings = []
    batch_size = 8

    for i in tqdm(range(0, len(texts), batch_size), desc=f"  {model_id.split('/')[-1]}"):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
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
            # last hidden state: (B, T, H)
            hidden = out.hidden_states[-1].float()

        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts).cpu().numpy()  # (B, H)
        embeddings.append(pooled)

    del model
    torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)  # (N, H)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel matrix (required for CKA)."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between representation matrices X (N, d1) and Y (N, d2).

    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Equivalent to the HSIC-based formulation with linear kernels.
    """
    # Gram matrices
    Kx = X @ X.T
    Ky = Y @ Y.T

    # Center
    Kx_c = center_kernel(Kx)
    Ky_c = center_kernel(Ky)

    # HSIC
    hsic_xy = np.sum(Kx_c * Ky_c)
    hsic_xx = np.sqrt(np.sum(Kx_c * Kx_c))
    hsic_yy = np.sqrt(np.sum(Ky_c * Ky_c))

    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return float(hsic_xy / (hsic_xx * hsic_yy))


def pca_eigenvalues(X: np.ndarray, n_components: int = PCA_COMPONENTS) -> np.ndarray:
    """
    Return the top-k eigenvalues (as fraction of total variance) of X.
    X shape: (N, D)
    """
    # Center
    X_c = X - X.mean(axis=0, keepdims=True)
    # Covariance (use SVD for numerical stability when D >> N)
    n = X_c.shape[0]
    _, s, _ = np.linalg.svd(X_c / np.sqrt(n - 1), full_matrices=False)
    eigenvalues = s ** 2
    # Normalise to sum=1 so spectra from different-dim spaces are comparable
    eigenvalues = eigenvalues / eigenvalues.sum()
    return eigenvalues[:n_components]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eigenspectra(eigenvalues_dict: dict, output_path: str):
    n = len(eigenvalues_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]

    for ax, (name, eigs), color in zip(axes, eigenvalues_dict.items(), colors):
        k = np.arange(1, len(eigs) + 1)
        ax.fill_between(k, eigs, alpha=0.25, color=color)
        ax.plot(k, eigs, color=color, linewidth=2)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Principal Component", fontsize=9)
        ax.set_ylabel("Fraction of Variance", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("PCA Eigenvalue Spectra (log scale)\nSimilar decay = similar intrinsic dimensionality", 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved eigenspectra → {output_path}")


def plot_cka_heatmap(cka_matrix: np.ndarray, names: list, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    n = len(names)

    im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Linear CKA (0=different, 1=identical)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = cka_matrix[i, j]
            text_color = "black" if 0.3 < val < 0.85 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    ax.set_title("Pairwise Linear CKA\n(audio vs text, and text vs text)", 
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CKA heatmap → {output_path}")


def plot_pca_scatter(embeddings_dict: dict, output_path: str):
    """2D PCA scatter of all models coloured by model, to visualise geometry."""
    from sklearn.decomposition import PCA as skPCA

    fig, axes = plt.subplots(1, len(embeddings_dict), figsize=(4 * len(embeddings_dict), 4))
    if len(embeddings_dict) == 1:
        axes = [axes]

    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]

    for ax, (name, X), color in zip(axes, embeddings_dict.items(), colors):
        pca = skPCA(n_components=2)
        Z = pca.fit_transform(X)
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.5, s=20, color=color, linewidths=0)
        var = pca.explained_variance_ratio_
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("2D PCA of Representations (each sample = one utterance)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA scatter → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Cross-modal representation analysis")
    p.add_argument("--n_samples", type=int, default=200,
                   help="Number of LibriSpeech samples to use (default: 200)")
    p.add_argument("--output_dir", type=str, default="./results",
                   help="Directory to save embeddings and plots")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip embedding extraction if cached .pkl exists")
    p.add_argument("--pca_components", type=int, default=PCA_COMPONENTS,
                   help="Number of PCA components for eigenvalue analysis")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading LibriSpeech ({args.n_samples} samples)…")
    print(f"{'='*60}")
    ds = load_dataset(
        DATASET_NAME,
        "clean",
        split=DATASET_SPLIT,
        streaming=False,
        trust_remote_code=True,
    )
    # Take a deterministic slice
    ds = ds.select(range(min(args.n_samples, len(ds))))
    texts = [s["text"].strip() for s in ds]
    print(f"Loaded {len(texts)} samples.")
    print(f"Example transcript: '{texts[0][:80]}…'")

    # ------------------------------------------------------------------
    # 2. Extract embeddings (one model at a time to save VRAM)
    # ------------------------------------------------------------------
    embeddings = {}

    for model_name, cfg in MODELS.items():
        cache_path = out_dir / f"embeddings_{model_name}.pkl"

        if args.skip_existing and cache_path.exists():
            print(f"\nLoading cached embeddings for {model_name}…")
            with open(cache_path, "rb") as f:
                embeddings[model_name] = pickle.load(f)
            continue

        print(f"\n{'='*60}")
        print(f"Extracting: {model_name} ({cfg['modality']})")
        print(f"{'='*60}")

        if cfg["modality"] == "audio":
            emb = extract_whisper_embeddings(cfg["hf_id"], list(ds), device)
        else:
            emb = extract_lm_embeddings(cfg["hf_id"], texts, device)

        print(f"  → Shape: {emb.shape}")
        embeddings[model_name] = emb

        with open(cache_path, "wb") as f:
            pickle.dump(emb, f)
        print(f"  → Cached to {cache_path}")

    # ------------------------------------------------------------------
    # 3. PCA eigenvalue spectra
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Computing PCA eigenvalue spectra…")
    print(f"{'='*60}")

    eigenvalues = {}
    for name, X in embeddings.items():
        print(f"  {name}: shape {X.shape}")
        eigenvalues[name] = pca_eigenvalues(X, n_components=args.pca_components)

    plot_eigenspectra(eigenvalues, str(out_dir / "eigenspectra.png"))

    # ------------------------------------------------------------------
    # 4. Pairwise CKA
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Computing pairwise CKA…")
    print(f"{'='*60}")

    names = list(embeddings.keys())
    n = len(names)
    cka_matrix = np.zeros((n, n))
    cka_results = {}

    for i in range(n):
        for j in range(n):
            score = linear_cka(embeddings[names[i]], embeddings[names[j]])
            cka_matrix[i, j] = score
            key = f"{names[i]} vs {names[j]}"
            cka_results[key] = score
            if i <= j:
                print(f"  CKA({names[i]}, {names[j]}) = {score:.4f}")

    plot_cka_heatmap(cka_matrix, names, str(out_dir / "cka_heatmap.png"))

    # ------------------------------------------------------------------
    # 5. 2D PCA scatter
    # ------------------------------------------------------------------
    try:
        from sklearn.decomposition import PCA  # noqa: F401 (just checking import)
        plot_pca_scatter(embeddings, str(out_dir / "pca_scatter.png"))
    except ImportError:
        print("scikit-learn not found, skipping 2D scatter (pip install scikit-learn)")

    # ------------------------------------------------------------------
    # 6. Save results JSON
    # ------------------------------------------------------------------
    results = {
        "n_samples": len(texts),
        "models": {k: v["hf_id"] for k, v in MODELS.items()},
        "embedding_shapes": {k: list(v.shape) for k, v in embeddings.items()},
        "cka_scores": cka_results,
        "eigenvalue_spectra": {k: v.tolist() for k, v in eigenvalues.items()},
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Pair':<45} {'CKA':>6}")
    print("-" * 53)
    for k, v in cka_results.items():
        if k.split(" vs ")[0] != k.split(" vs ")[1]:
            print(f"{k:<45} {v:>6.4f}")

    print(f"\nAll outputs in: {out_dir.resolve()}")
    print("\nInterpretation guide:")
    print("  CKA ~1.0 : nearly identical representational geometry")
    print("  CKA ~0.5 : moderate similarity")
    print("  CKA ~0.0 : no shared structure")
    print("\n  Cross-modal (whisper vs LLM) CKA > ~0.3 would be noteworthy evidence")
    print("  for the Platonic Representation Hypothesis.")


if __name__ == "__main__":
    main()
