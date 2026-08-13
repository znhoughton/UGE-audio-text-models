#!/usr/bin/env python3
"""
Whisper-Kaldi-LLM Variance Decomposition via INLP + Nonlinear Verification

Decomposes Whisper embedding variance into four components:
  - Kaldi-only    : variance in the linear Kaldi subspace but not the linear LLM subspace
  - LLM-only      : variance in the linear LLM subspace but not the linear Kaldi subspace
  - Shared        : variance in both linear subspaces
  - Neither       : variance in neither linear subspace

Method:
  Linear deflation (INLP-style) is done in one shot using Ridge regression.
  Because the OLS residual is orthogonal to X_target by construction, no
  iteration is needed — a second linear probe on the residual explains 0%.
  Both orderings are run to recover the shared component algebraically:

  Ordering 1 (Kaldi first):
    1. Ridge(X_kaldi → X_whisper) → subtract → res_k
    2. Ridge(X_llm   → res_k)     → subtract → res_kl

  Ordering 2 (LLM first):
    1. Ridge(X_llm   → X_whisper) → subtract → res_l
    2. Ridge(X_kaldi → res_l)     → subtract → res_lk

  Four components:
    kaldi_only = V_kaldi_given_llm  (ordering 2, step 2)
    llm_only   = V_llm_given_kaldi  (ordering 1, step 2)
    shared     = V_kaldi_total - kaldi_only  (== V_llm_total - llm_only)
    neither    = 1 - kaldi_only - llm_only - shared

  Nonlinear verification:
    MLP probes are trained on the "neither" residual to test whether any
    nonlinear Kaldi or LLM structure survived the linear deflation.
    Low R² (<0.05) confirms the linear subspace captured essentially all
    shared structure; high R² is reported and discussed.

Usage (word-level):
  python projection_analysis.py --granularity word \\
      --embeddings_dir /dpluth-data \\
      --output_dir ./word

Usage (phone-level):
  python projection_analysis.py --granularity phone \\
      --embeddings_dir /dpluth-data \\
      --output_dir ./phone
"""

import argparse
import csv
import gc
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("projection")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHISPER_ENCODERS = [
    "whisper-base-enc",
    "whisper-small-enc",
    "whisper-medium-enc",
    "whisper-large-enc",
    "whisper-large-v2-enc",
    "whisper-large-v1-enc",
]

WHISPER_DECODERS = [
    "whisper-base-dec",
    "whisper-small-dec",
    "whisper-medium-dec",
    "whisper-large-dec",
    "whisper-large-v2-dec",
    "whisper-large-v1-dec",
]

WHISPER_ALL = WHISPER_ENCODERS + WHISPER_DECODERS

KALDI_MODELS = [
    "kaldi-librispeech",
    "kaldi-librispeech-penult",
    "kaldi-librispeech-antepen",
]

LLM_MODELS = [
    "babylm-125m",
    "opt-125m",
    "babylm-350m",
    "babylm-1.3b",
    "pythia-160m",
    "olmo-7b",
    "pythia-6.9b",
]

# Alphas tried by val-split CV (log-spaced)
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

# Rows processed per chunk in Ridge helpers — keeps peak extra RAM to ~1.6 GB
# even when X_target is OLMo (4096-dim, 885k rows = 14.5 GB in float16)
RIDGE_CHUNK = 50_000


# ---------------------------------------------------------------------------
# Chunked Ridge helpers — no full float32 copy of X_target ever created
# ---------------------------------------------------------------------------

def _gram_and_cross(X_target, X_residual, mask, chunk_size=RIDGE_CHUNK):
    """Accumulate Gram (X.T @ X) and cross-cov (X.T @ Y) in float64 chunks.

    Inputs may be float16; each chunk is upcasted to float64 on the fly so
    the Gram matrix is numerically stable even for near-singular embeddings.
    Peak extra RAM per iteration: chunk_size × (d_t + d_w) × 8 bytes ≈ 1.6 GB.
    """
    d_t = X_target.shape[1]
    d_w = X_residual.shape[1]
    gram  = np.zeros((d_t, d_t), dtype=np.float64)
    cross = np.zeros((d_t, d_w), dtype=np.float64)
    for start in range(0, len(X_target), chunk_size):
        end = min(start + chunk_size, len(X_target))
        m   = mask[start:end]
        if not m.any():
            continue
        X_c = X_target[start:end][m].astype(np.float64)
        Y_c = X_residual[start:end][m].astype(np.float64)
        gram  += X_c.T @ X_c
        cross += X_c.T @ Y_c
        del X_c, Y_c
    gc.collect()
    return gram, cross


def _val_r2_chunked(W, X_target, X_residual, val_mask, chunk_size=RIDGE_CHUNK):
    """Compute R² on the validation set in float32 chunks."""
    preds, trues = [], []
    for start in range(0, len(X_target), chunk_size):
        end = min(start + chunk_size, len(X_target))
        m   = val_mask[start:end]
        if not m.any():
            continue
        preds.append(X_target[start:end][m].astype(np.float32) @ W)
        trues.append(X_residual[start:end][m].astype(np.float32))
    if not preds:
        return 0.0
    return r2_score(np.concatenate(trues), np.concatenate(preds))


def _predict_subtract_chunked(W, X_target, X_residual, chunk_size=RIDGE_CHUNK):
    """Compute (X_residual - X_target @ W) in float32 chunks, stored as float16.

    Avoids ever creating a full float32 copy of X_target or X_residual.
    """
    result = np.empty(X_residual.shape, dtype=np.float16)
    for start in range(0, len(X_target), chunk_size):
        end = min(start + chunk_size, len(X_target))
        X_c = X_target[start:end].astype(np.float32)
        Y_c = X_residual[start:end].astype(np.float32)
        result[start:end] = (Y_c - X_c @ W).astype(np.float16)
        del X_c, Y_c
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Linear deflation (INLP-style, one shot)
# ---------------------------------------------------------------------------

def linear_deflation(
    X_whisper_residual: np.ndarray,
    X_target: np.ndarray,
    val_mask: np.ndarray,
) -> tuple:
    """
    Fit Ridge regression (X_target → X_whisper_residual) on the training split,
    then subtract predictions from the full dataset.

    Because the OLS residual is orthogonal to X_target by construction, one step
    removes all linearly-accessible variance — no iteration required.

    Memory strategy (matches phone_level_analysis.py approach):
      - Embeddings are stored as float16 (halves baseline RAM footprint)
      - Gram matrix and cross-covariance are accumulated in float64 chunks so
        no full float32 copy of X_target (14.5 GB for OLMo) is ever created
      - Alpha selected via val split (avoids RidgeCV LOO-SVD which needs an
        O(n×p) U matrix — 11.6 GB for OLMo × 4 calls per Whisper model)
      - Residual stored as float16 to halve persistent memory between models

    Returns:
        residual     (np.ndarray, float16): X_whisper_residual - Ridge prediction
        r2_val       (float):               R² on held-out validation split
        alpha_chosen (float):               Ridge alpha selected by val-split CV
    """
    train_mask = ~val_mask
    d_t = X_target.shape[1]

    gram, cross = _gram_and_cross(X_target, X_whisper_residual, train_mask)

    best_alpha, best_r2, best_W = RIDGE_ALPHAS[0], -np.inf, None
    for alpha in RIDGE_ALPHAS:
        A = gram + alpha * np.eye(d_t, dtype=np.float64)
        W = np.linalg.solve(A, cross).astype(np.float32)  # (d_t, d_w)
        r2 = _val_r2_chunked(W, X_target, X_whisper_residual, val_mask)
        if r2 > best_r2:
            best_r2, best_alpha, best_W = r2, alpha, W
        del W
    del gram, cross
    gc.collect()

    residual = _predict_subtract_chunked(best_W, X_target, X_whisper_residual)
    del best_W
    gc.collect()

    return residual, float(best_r2), float(best_alpha)


# ---------------------------------------------------------------------------
# MLP probe (used only for nonlinear verification)
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 1024, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.GELU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_probe(
    X_in_tr: np.ndarray, X_out_tr: np.ndarray,
    X_in_val: np.ndarray, X_out_val: np.ndarray,
    hidden_dim: int = 1024, n_layers: int = 3, dropout: float = 0.1,
    lr: float = 1e-3, batch_size: int = 2048,
    max_epochs: int = 150, patience: int = 10,
    device: str = "cuda",
) -> MLPProbe:
    """Train MLP with early stopping on validation MSE.

    Full dataset is GPU-resident for the duration of training; shuffling uses
    torch.randperm on the GPU. bfloat16 AMP gives ~2x throughput on A100/H100.
    """
    probe = MLPProbe(X_in_tr.shape[1], X_out_tr.shape[1],
                     hidden_dim, n_layers, dropout).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    t_in  = torch.from_numpy(X_in_tr.astype(np.float32)).to(device)
    t_out = torch.from_numpy(X_out_tr.astype(np.float32)).to(device)
    v_in  = torch.from_numpy(X_in_val.astype(np.float32)).to(device)
    v_out = torch.from_numpy(X_out_val.astype(np.float32)).to(device)
    N_tr  = t_in.shape[0]

    best_val, best_state, no_improve = float("inf"), None, 0
    probe.train()
    for epoch in range(max_epochs):
        perm = torch.randperm(N_tr, device=device)
        for start in range(0, N_tr, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = t_in[idx], t_out[idx]
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = nn.functional.mse_loss(probe(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        probe.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            val_loss = nn.functional.mse_loss(probe(v_in), v_out).item()
        probe.train()

        sched.step(val_loss)
        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            logger.debug(f"    Early stop at epoch {epoch + 1}")
            break

    probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    probe.eval()
    return probe


def predict_batched(probe: MLPProbe, X: np.ndarray,
                    batch_size: int = 4096, device: str = "cuda") -> np.ndarray:
    probe.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
            chunks.append(probe(xb).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² across all output dimensions pooled."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0, keepdims=True)) ** 2).sum())
    return 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def nonlinear_verification(
    X_residual: np.ndarray,
    X_target: np.ndarray,
    val_mask: np.ndarray,
    device: str = "cuda",
    **probe_kwargs,
) -> float:
    """
    Train an MLP (X_target → X_residual) and return R² on the validation split.

    Intended to be called on the "neither" residual after linear deflation.
    Low R² confirms the linear subspace captured all shared structure.
    High R² indicates surviving nonlinear structure and should be reported.
    """
    train_mask = ~val_mask
    probe = train_probe(
        X_target[train_mask], X_residual[train_mask],
        X_target[val_mask],   X_residual[val_mask],
        device=device, **probe_kwargs,
    )
    pred_val = predict_batched(probe, X_target[val_mask], device=device)
    r2 = r2_score(X_residual[val_mask], pred_val)

    del probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(r2)


# ---------------------------------------------------------------------------
# Venn decomposition
# ---------------------------------------------------------------------------

def venn_decomposition(
    X_whisper: np.ndarray,
    X_kaldi: np.ndarray,
    X_llm: np.ndarray,
    val_mask: np.ndarray,
    device: str = "cuda",
    **probe_kwargs,
) -> dict:
    """
    Run both orderings of linear deflation and recover four variance components,
    then verify with nonlinear MLP probes on the neither residual.

    From two orderings:
      Ordering 1 — Kaldi first:
        V_kaldi_total = Kaldi-only + Shared
        V_llm_given_k = LLM-only
      Ordering 2 — LLM first:
        V_llm_total     = LLM-only + Shared
        V_kaldi_given_l = Kaldi-only

    Therefore:
      kaldi_only = V_kaldi_given_l
      llm_only   = V_llm_given_k
      shared     = V_kaldi_total - kaldi_only  (== V_llm_total - llm_only)
      neither    = 1 - kaldi_only - llm_only - shared
    """
    # Cast to float32 before squaring/summing — float16 overflows at 65504 and
    # loses precision when accumulating 885k values
    total_var = float((X_whisper.astype(np.float32) ** 2).sum())

    def vfrac(arr):
        return float((arr.astype(np.float32) ** 2).sum() / total_var)

    # ---- Ordering 1: Kaldi → LLM ----------------------------------------
    logger.info("  [Ordering 1] Ridge: Kaldi → Whisper...")
    res_k, r2_k, alpha_k = linear_deflation(X_whisper, X_kaldi, val_mask)
    logger.info(f"    R²={r2_k:.4f}  alpha={alpha_k}")
    V_kaldi_total = 1.0 - vfrac(res_k)
    vfrac_res_k   = vfrac(res_k)

    logger.info("  [Ordering 1] Ridge: LLM → Kaldi residual...")
    res_kl, r2_kl, alpha_kl = linear_deflation(res_k, X_llm, val_mask)
    logger.info(f"    R²={r2_kl:.4f}  alpha={alpha_kl}")
    del res_k; gc.collect()

    V_llm_given_k  = vfrac_res_k - vfrac(res_kl)
    V_neither_ord1 = vfrac(res_kl)

    # ---- Ordering 2: LLM → Kaldi ----------------------------------------
    logger.info("  [Ordering 2] Ridge: LLM → Whisper...")
    res_l, r2_l, alpha_l = linear_deflation(X_whisper, X_llm, val_mask)
    logger.info(f"    R²={r2_l:.4f}  alpha={alpha_l}")
    V_llm_total = 1.0 - vfrac(res_l)
    vfrac_res_l = vfrac(res_l)

    logger.info("  [Ordering 2] Ridge: Kaldi → LLM residual...")
    res_lk, r2_lk, alpha_lk = linear_deflation(res_l, X_kaldi, val_mask)
    logger.info(f"    R²={r2_lk:.4f}  alpha={alpha_lk}")
    del res_l; gc.collect()

    V_kaldi_given_l = vfrac_res_l - vfrac(res_lk)
    V_neither_ord2  = vfrac(res_lk)

    # ---- Four components -------------------------------------------------
    kaldi_only = V_kaldi_given_l
    llm_only   = V_llm_given_k
    shared     = V_kaldi_total - kaldi_only
    neither    = (V_neither_ord1 + V_neither_ord2) / 2.0

    shared_check = V_llm_total - llm_only
    if abs(shared - shared_check) > 0.02:
        logger.warning(f"  Shared inconsistency: {shared:.4f} vs {shared_check:.4f}")
    if abs(V_neither_ord1 - V_neither_ord2) > 0.02:
        logger.warning(f"  Neither inconsistency: ord1={V_neither_ord1:.4f} ord2={V_neither_ord2:.4f}")

    # ---- Nonlinear verification on neither residual ----------------------
    neither_residual = (res_kl + res_lk) / 2.0
    del res_kl, res_lk; gc.collect()

    logger.info("  [Verification] MLP probe: Kaldi → neither residual...")
    nlr2_kaldi = nonlinear_verification(neither_residual, X_kaldi, val_mask,
                                        device=device, **probe_kwargs)
    logger.info(f"    nonlinear R²(Kaldi)={nlr2_kaldi:.4f}")

    logger.info("  [Verification] MLP probe: LLM → neither residual...")
    nlr2_llm = nonlinear_verification(neither_residual, X_llm, val_mask,
                                      device=device, **probe_kwargs)
    logger.info(f"    nonlinear R²(LLM)={nlr2_llm:.4f}")
    del neither_residual; gc.collect()

    if nlr2_kaldi > 0.05:
        logger.warning(f"  Nonlinear Kaldi structure remains in neither residual: R²={nlr2_kaldi:.4f}")
    if nlr2_llm > 0.05:
        logger.warning(f"  Nonlinear LLM structure remains in neither residual: R²={nlr2_llm:.4f}")

    return {
        "kaldi_only": round(kaldi_only, 5),
        "llm_only":   round(llm_only, 5),
        "shared":     round(shared, 5),
        "neither":    round(neither, 5),
        # Nonlinear verification
        "nlr2_kaldi": round(nlr2_kaldi, 5),
        "nlr2_llm":   round(nlr2_llm, 5),
        # Raw quantities for transparency / debugging
        "V_kaldi_total":     round(V_kaldi_total, 5),
        "V_llm_total":       round(V_llm_total, 5),
        "V_kaldi_given_llm": round(V_kaldi_given_l, 5),
        "V_llm_given_kaldi": round(V_llm_given_k, 5),
        "neither_ordering1": round(V_neither_ord1, 5),
        "neither_ordering2": round(V_neither_ord2, 5),
        "shared_check":      round(shared_check, 5),
        "linear_r2": {
            "ord1_kaldi":      round(r2_k, 5),
            "ord1_llm_given_k": round(r2_kl, 5),
            "ord2_llm":        round(r2_l, 5),
            "ord2_kaldi_given_l": round(r2_lk, 5),
        },
        "ridge_alphas": {
            "ord1_kaldi":      alpha_k,
            "ord1_llm_given_k": alpha_kl,
            "ord2_llm":        alpha_l,
            "ord2_kaldi_given_l": alpha_lk,
        },
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embedding(embeddings_dir: Path, granularity: str, model_name: str) -> np.ndarray:
    subdir = "WordData"  if granularity == "word"  else "PhoneData"
    prefix = "word"      if granularity == "word"  else "phone"
    path = embeddings_dir / subdir / f"{prefix}_embeddings_{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Embedding not found: {path}")
    logger.info(f"  Loading {path}")
    with open(path, "rb") as f:
        emb = pickle.load(f)
    return emb.astype(np.float16)  # halves persistent RAM; upcasted per-chunk during computation


def build_llm_target(embeddings_dir: Path, granularity: str,
                     llm_models: list, pca_dims: int, seed: int) -> np.ndarray:
    """
    If one LLM: return its embeddings directly (centered).
    If multiple LLMs: compute a shared subspace via PCA on pooled embeddings,
    return projection onto top pca_dims PCs.
    """
    if len(llm_models) == 1:
        X = load_embedding(embeddings_dir, granularity, llm_models[0])
        return X - X.mean(0)

    logger.info(f"Building shared LLM subspace from {len(llm_models)} models "
                f"(PCA → {pca_dims} dims)")
    arrays = []
    for m in llm_models:
        try:
            e = load_embedding(embeddings_dir, granularity, m).astype(np.float64)
            e -= e.mean(0)
            e /= (np.linalg.norm(e, axis=1).mean() + 1e-8)
            arrays.append(e)
        except FileNotFoundError as ex:
            logger.warning(f"  {ex} — skipping")

    if not arrays:
        raise RuntimeError("No LLM embeddings found")

    stacked = np.concatenate(arrays, axis=1)
    del arrays; gc.collect()

    stacked -= stacked.mean(0)
    stacked /= np.sqrt(stacked.shape[0] - 1)

    from sklearn.utils.extmath import randomized_svd
    _, _, Vt = randomized_svd(stacked, n_components=pca_dims, random_state=seed)
    del stacked; gc.collect()

    arrays2 = []
    for m in llm_models:
        try:
            e = load_embedding(embeddings_dir, granularity, m).astype(np.float64)
            e -= e.mean(0)
            e /= (np.linalg.norm(e, axis=1).mean() + 1e-8)
            arrays2.append(e)
        except FileNotFoundError:
            pass
    stacked2 = np.concatenate(arrays2, axis=1).astype(np.float32)
    del arrays2; gc.collect()

    projection = stacked2 @ Vt.T.astype(np.float32)
    del stacked2; gc.collect()
    logger.info(f"  Shared LLM target shape: {projection.shape}")
    return projection


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Whisper-Kaldi-LLM variance decomposition (INLP + nonlinear verification)"
    )
    p.add_argument("--granularity", choices=["word", "phone"], required=True)
    p.add_argument("--embeddings_dir", type=Path, default=Path("/dpluth-data"),
                   help="Root dir containing WordData/ and PhoneData/ pkl files")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where to save results (default: ./<granularity>/)")
    p.add_argument("--whisper_models", nargs="+", default=WHISPER_ALL)
    p.add_argument("--kaldi_model", default="kaldi-librispeech", choices=KALDI_MODELS)
    p.add_argument("--llm_models", nargs="+", default=["olmo-7b"],
                   help="LLM(s) to use. Defaults to olmo-7b. Multiple → shared PCA subspace.")
    p.add_argument("--llm_pca_dims", type=int, default=256,
                   help="PCA dims for shared LLM subspace when >1 LLM given")
    # MLP verification hyperparameters
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction held out for R² evaluation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = args.output_dir or Path(args.granularity)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"projection_run_{ts}.log"
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Projection analysis — {args.granularity}-level")
    logger.info(f"Arguments: {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load shared inputs (Kaldi + LLM) once; reuse across Whisper models
    # ------------------------------------------------------------------
    logger.info(f"Loading Kaldi: {args.kaldi_model}")
    X_kaldi = load_embedding(args.embeddings_dir, args.granularity, args.kaldi_model)
    X_kaldi -= X_kaldi.mean(0)
    logger.info(f"  Kaldi shape: {X_kaldi.shape}")

    X_llm = build_llm_target(args.embeddings_dir, args.granularity,
                              args.llm_models, args.llm_pca_dims, args.seed)

    N = len(X_kaldi)
    rng = np.random.default_rng(args.seed)
    val_mask = rng.random(N) < args.val_frac
    logger.info(f"N={N:,}  train={int((~val_mask).sum()):,}  val={int(val_mask.sum()):,}")

    probe_kwargs = dict(
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        dropout=args.dropout, lr=args.lr, batch_size=args.batch_size,
        max_epochs=args.max_epochs, patience=args.patience,
    )

    # ------------------------------------------------------------------
    # Run per Whisper model
    # ------------------------------------------------------------------
    all_results = {}

    for whisper_name in args.whisper_models:
        logger.info("=" * 60)
        logger.info(f"Whisper model: {whisper_name}")
        logger.info("=" * 60)

        try:
            X_w = load_embedding(args.embeddings_dir, args.granularity, whisper_name)
        except FileNotFoundError as e:
            logger.warning(f"  {e} — skipping")
            continue

        logger.info(f"  Whisper shape: {X_w.shape}")
        X_w -= X_w.mean(0)

        result = venn_decomposition(
            X_w, X_kaldi, X_llm, val_mask,
            device=device, **probe_kwargs,
        )

        logger.info(f"  --- {whisper_name} summary ---")
        logger.info(f"  Kaldi-only   : {result['kaldi_only']:.4f}")
        logger.info(f"  LLM-only     : {result['llm_only']:.4f}")
        logger.info(f"  Shared       : {result['shared']:.4f}")
        logger.info(f"  Neither      : {result['neither']:.4f}")
        logger.info(f"  (sum={result['kaldi_only']+result['llm_only']+result['shared']+result['neither']:.4f})")
        logger.info(f"  nlr2 Kaldi   : {result['nlr2_kaldi']:.4f}")
        logger.info(f"  nlr2 LLM     : {result['nlr2_llm']:.4f}")

        all_results[whisper_name] = result

        del X_w
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out_path = output_dir / "projection_results.json"
    payload = {
        "granularity":  args.granularity,
        "kaldi_model":  args.kaldi_model,
        "llm_models":   args.llm_models,
        "llm_pca_dims": args.llm_pca_dims,
        "method":       "ridge_inlp_plus_mlp_verification",
        "ridge_alphas_tried": RIDGE_ALPHAS,
        "results":      all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results JSON → {out_path}")

    # ------------------------------------------------------------------
    # Save per-model CSVs + summary CSV
    # ------------------------------------------------------------------
    COMPONENTS = ["kaldi_only", "llm_only", "shared", "neither",
                  "nlr2_kaldi", "nlr2_llm",
                  "V_kaldi_total", "V_llm_total"]

    for whisper_name, result in all_results.items():
        csv_path = output_dir / f"{whisper_name}_decomposition.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["component", "value"])
            for comp in COMPONENTS:
                writer.writerow([comp, result[comp]])
        logger.info(f"CSV → {csv_path}")

    summary_path = output_dir / "decomposition_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "modality"] + COMPONENTS)
        for whisper_name, result in all_results.items():
            modality = "encoder" if whisper_name.endswith("-enc") else "decoder"
            writer.writerow(
                [whisper_name, modality] + [result[c] for c in COMPONENTS]
            )
    logger.info(f"Summary CSV → {summary_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
