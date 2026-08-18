#!/usr/bin/env python3
"""
Whisper-Kaldi-LLM Variance Decomposition via Ridge Regression + Nonlinear Verification

Decomposes Whisper embedding variance into four components:
  - Kaldi-only    : variance in the linear Kaldi subspace but not the linear LLM subspace
  - LLM-only      : variance in the linear LLM subspace but not the linear Kaldi subspace
  - Shared        : variance in both linear subspaces
  - Neither       : variance in neither linear subspace

Method (regress Whisper ON the target — target is the predictor):
  We regress Whisper ON the target (Kaldi or LLM): X_whisper ≈ X_target · W,
  and subtract the prediction. The residual is the part of Whisper NOT linearly
  predictable from the target:

      res = X_whisper − X_target · W

  Whisper must be the response so that its total variance is the denominator
  every fraction is measured against — that is what makes the four components a
  decomposition of *Whisper's* variance. Because it is OLS/ridge regression, the
  residual is (near-)orthogonal to the target in one shot; a second linear probe
  on the residual explains ~0.

  Why this is confound-free (no PCA / rank-matching needed):
    HELD-OUT VARIANCE. W is fit on the train split; every variance fraction is
    measured on the val split only. In-sample, a p-dim predictor can spuriously
    subtract ~p/n of any non-target Whisper direction (a finite-sample overlap
    with the target's column space, NOT true structure). Out-of-sample that
    overlap does not transfer, so the reported (held-out) numbers remove it
    entirely — the wrongly-subtracted fraction is the in-sample−held-out gap,
    ~0.07% at our n, and ~0 on val. Ridge shrinks the projection, subtracting
    even less. Optional --null_check permutes target rows to confirm the removed
    held-out variance collapses to ~0.

  Both orderings are run to recover the shared component algebraically:

  Ordering 1 (Kaldi first):
    1. regress Whisper on Kaldi, subtract → res_k
    2. regress res_k   on LLM,   subtract → res_kl

  Ordering 2 (LLM first):
    1. regress Whisper on LLM,   subtract → res_l
    2. regress res_l   on Kaldi, subtract → res_lk

  Four components (fractions of Whisper's held-out variance):
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

# Alphas tried by val-split CV (log-spaced) — regularizes the target→Whisper fit
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

# Rows processed per chunk in the accumulators — keeps peak extra RAM bounded
# even when a target is OLMo (4096-dim, 885k rows = 14.5 GB in float16)
RIDGE_CHUNK = 50_000


# ---------------------------------------------------------------------------
# Chunked Ridge helpers — predictor is the target (Kaldi/LLM); no full float32
# copy of X_target is ever created, so OLMo (4096-dim, 14.5 GB f16) is safe.
# ---------------------------------------------------------------------------

def _gram_and_cross(X_target, X_whisper, mask, chunk_size=RIDGE_CHUNK):
    """Accumulate Gram G = Xtᵀ Xt (d_t × d_t) and cross C = Xtᵀ Xw (d_t × d_w)
    over masked rows, in float64. Peak extra RAM per chunk ≈ chunk_size ×
    (d_t + d_w) × 8 bytes.
    """
    d_t = X_target.shape[1]
    d_w = X_whisper.shape[1]
    G = np.zeros((d_t, d_t), dtype=np.float64)
    C = np.zeros((d_t, d_w), dtype=np.float64)
    for start in range(0, len(X_target), chunk_size):
        end = min(start + chunk_size, len(X_target))
        m   = mask[start:end]
        if not m.any():
            continue
        Xt = X_target[start:end][m].astype(np.float64)
        Xw = X_whisper[start:end][m].astype(np.float64)
        G += Xt.T @ Xt
        C += Xt.T @ Xw
        del Xt, Xw
    gc.collect()
    return G, C


def _val_r2_chunked(W, X_target, X_whisper, val_mask, chunk_size=RIDGE_CHUNK):
    """R² of predicting Whisper FROM the target (Whisper ≈ Xt·W) on the val split.

    This is the held-out variance fraction the fit removes — the honest,
    overfitting-free measure of how much of Whisper lies in the target space.
    """
    preds, trues = [], []
    for start in range(0, len(X_target), chunk_size):
        end = min(start + chunk_size, len(X_target))
        m   = val_mask[start:end]
        if not m.any():
            continue
        preds.append(X_target[start:end][m].astype(np.float32) @ W)
        trues.append(X_whisper[start:end][m].astype(np.float32))
    if not preds:
        return 0.0
    return r2_score(np.concatenate(trues), np.concatenate(preds))


def _predict_subtract_chunked(W, X_target, X_whisper, chunk_size=RIDGE_CHUNK):
    """Compute residual = X_whisper − X_target · W in float32 chunks, stored
    as float16. Never creates a full float32 copy of X_target or X_whisper.
    """
    result = np.empty(X_whisper.shape, dtype=np.float16)
    for start in range(0, len(X_target), chunk_size):
        end = min(start + chunk_size, len(X_target))
        Xt = X_target[start:end].astype(np.float32)
        Xw = X_whisper[start:end].astype(np.float32)
        result[start:end] = (Xw - Xt @ W).astype(np.float16)
        del Xt, Xw
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Linear deflation (regress Whisper ON the target, subtract the projection)
# ---------------------------------------------------------------------------

def linear_deflation(
    X_whisper_residual: np.ndarray,
    X_target: np.ndarray,
    val_mask: np.ndarray,
) -> tuple:
    """
    Fit Ridge  Whisper ≈ X_target · W  on the train split, then subtract the
    prediction from the full dataset. The residual is the part of Whisper NOT
    linearly predictable from the target.

    Direction is Kaldi/LLM → Whisper (target is the predictor): to decompose
    Whisper's variance, Whisper must be the response so its total variance is
    the denominator every fraction is measured against.

    Guarantees / properties:
      - The most non-target variance that can be wrongly subtracted is
        rank(X_target)/n (in-sample), and ~0 out-of-sample. Held-out R²
        (see venn_decomposition) is what we report, so this is confound-free.
      - Alpha is selected by held-out R² (not RidgeCV LOO-SVD, which needs an
        O(n×p) U matrix). Ridge shrinks the projection, so it subtracts even
        less non-target variance than OLS.
      - Gram/cross accumulated in float64 chunks; residual stored as float16.

    Returns:
        residual  (np.ndarray, float16): X_whisper − X_target · W
        r2_val    (float): held-out R² of the Whisper ≈ Xt·W fit (variance removed)
        alpha     (float): Ridge alpha selected by val-split CV
        r2_train  (float): in-sample R² of the SAME fit. The gap r2_train−r2_val
                           is the collateral estimate — variance wrongly subtracted
                           from non-target Whisper dims (~rank(target)/n, →0 out of
                           sample). No re-fit: reuses the chosen coefficients.
    """
    train_mask = ~val_mask
    d_t = X_target.shape[1]

    G, C = _gram_and_cross(X_target, X_whisper_residual, train_mask)

    best_alpha, best_r2, best_W = RIDGE_ALPHAS[0], -np.inf, None
    for alpha in RIDGE_ALPHAS:
        W = np.linalg.solve(G + alpha * np.eye(d_t, dtype=np.float64), C)  # (d_t, d_w)
        r2 = _val_r2_chunked(W.astype(np.float32), X_target, X_whisper_residual, val_mask)
        if r2 > best_r2:
            best_r2, best_alpha, best_W = r2, alpha, W
        del W
    del G, C
    gc.collect()

    Wf = best_W.astype(np.float32)
    # In-sample R² reuses the chosen coefficients — one extra eval, no re-fit.
    r2_train = _val_r2_chunked(Wf, X_target, X_whisper_residual, train_mask)
    residual = _predict_subtract_chunked(Wf, X_target, X_whisper_residual)
    del best_W, Wf
    gc.collect()

    return residual, float(best_r2), float(best_alpha), float(r2_train)


def null_floor(X_whisper, X_target, val_mask, seed=0):
    """Chance-level check: permute the target rows (breaking correspondence)
    and run the same deflation. In the forward direction the removed held-out
    variance should collapse to ~0, confirming no non-target structure is
    spuriously subtracted.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_target))
    res, r2, _, _ = linear_deflation(X_whisper, X_target[perm], val_mask)
    val_idx = val_mask
    total = float((X_whisper[val_idx].astype(np.float32) ** 2).sum())
    removed = 1.0 - float((res[val_idx].astype(np.float32) ** 2).sum() / total)
    del res; gc.collect()
    return removed, float(r2)


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

    del probe, pred_val
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
    # Variance fractions are measured on the VAL split only, so extra target
    # dimensions that only help in-sample contribute nothing (held-out lever).
    # Cast to float32 before squaring/summing — float16 overflows at 65504.
    val_idx = val_mask
    total_var = float((X_whisper[val_idx].astype(np.float32) ** 2).sum())

    def vfrac(arr):
        return float((arr[val_idx].astype(np.float32) ** 2).sum() / total_var)

    def _log_fit(tag, r2_val, r2_tr, alpha):
        logger.info(f"    {tag}: held-out R²={r2_val:.4f}  in-sample R²={r2_tr:.4f}  "
                    f"collateral gap={r2_tr - r2_val:+.4f}  alpha={alpha}")

    # ---- Ordering 1: Kaldi → LLM ----------------------------------------
    logger.info("  [Ordering 1] Regress Whisper on Kaldi, subtract...")
    res_k, r2_k, alpha_k, insr2_k = linear_deflation(X_whisper, X_kaldi, val_mask)
    _log_fit("Kaldi", r2_k, insr2_k, alpha_k)
    V_kaldi_total = 1.0 - vfrac(res_k)
    vfrac_res_k   = vfrac(res_k)

    logger.info("  [Ordering 1] Regress Kaldi residual on LLM, subtract...")
    res_kl, r2_kl, alpha_kl, insr2_kl = linear_deflation(res_k, X_llm, val_mask)
    _log_fit("LLM|Kaldi", r2_kl, insr2_kl, alpha_kl)
    del res_k; gc.collect()

    V_llm_given_k  = vfrac_res_k - vfrac(res_kl)
    V_neither_ord1 = vfrac(res_kl)

    # ---- Ordering 2: LLM → Kaldi ----------------------------------------
    logger.info("  [Ordering 2] Regress Whisper on LLM, subtract...")
    res_l, r2_l, alpha_l, insr2_l = linear_deflation(X_whisper, X_llm, val_mask)
    _log_fit("LLM", r2_l, insr2_l, alpha_l)
    V_llm_total = 1.0 - vfrac(res_l)
    vfrac_res_l = vfrac(res_l)

    logger.info("  [Ordering 2] Regress LLM residual on Kaldi, subtract...")
    res_lk, r2_lk, alpha_lk, insr2_lk = linear_deflation(res_l, X_kaldi, val_mask)
    _log_fit("Kaldi|LLM", r2_lk, insr2_lk, alpha_lk)
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
        # In-sample R² of each fit (reported alongside held-out linear_r2)
        "insample_r2": {
            "ord1_kaldi":      round(insr2_k, 5),
            "ord1_llm_given_k": round(insr2_kl, 5),
            "ord2_llm":        round(insr2_l, 5),
            "ord2_kaldi_given_l": round(insr2_lk, 5),
        },
        # Collateral estimate = in-sample − held-out (variance wrongly subtracted
        # from non-target Whisper dims; ~rank(target)/n, →0 out of sample)
        "collateral_gap": {
            "ord1_kaldi":      round(insr2_k - r2_k, 5),
            "ord1_llm_given_k": round(insr2_kl - r2_kl, 5),
            "ord2_llm":        round(insr2_l - r2_l, 5),
            "ord2_kaldi_given_l": round(insr2_lk - r2_lk, 5),
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
# Grouped train/val split (prevents leakage from repeated items)
# ---------------------------------------------------------------------------

def load_group_ids(embeddings_dir: Path, granularity: str, group_by: str,
                   expected_n: int, records_path: Path = None) -> np.ndarray:
    """Load one group label per embedding row from the *_records.json file.

    Records are row-aligned with the embeddings (embedding[i] ↔ records[i]).
    Grouping by e.g. utt_id keeps every utterance entirely in train or val, so
    the same item cannot leak across the split (which lets a high-capacity MLP
    memorize a lookup table and report a spurious R²≈1).
    """
    subdir = "WordData" if granularity == "word" else "PhoneData"
    prefix = "word"      if granularity == "word" else "phone"
    path = records_path or (embeddings_dir / subdir / f"{prefix}_records.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Records file for grouped split not found: {path}. "
            f"Pass --records_path, or --group_by none for a random split."
        )
    logger.info(f"  Loading group ids from {path} (group_by='{group_by}')")
    with open(path) as f:
        records = json.load(f)
    if len(records) != expected_n:
        raise ValueError(
            f"Records length {len(records):,} != embedding rows {expected_n:,}; "
            f"cannot align a grouped split. Check that records and embeddings match."
        )
    if group_by not in records[0]:
        raise KeyError(
            f"group_by='{group_by}' not in record fields {list(records[0].keys())}"
        )
    return np.array([str(rec[group_by]) for rec in records])


def grouped_val_mask(groups: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Assign whole groups to val until ~val_frac of ROWS are held out.

    Deterministic given seed. No group spans train and val.
    """
    rng = np.random.default_rng(seed)
    uniq, cnts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    target = val_frac * len(groups)
    val_groups, acc = set(), 0
    for i in order:
        if acc >= target:
            break
        val_groups.add(uniq[i])
        acc += cnts[i]
    return np.isin(groups, np.fromiter(val_groups, dtype=uniq.dtype))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Whisper-Kaldi-LLM variance decomposition (Whisper-on-target ridge + nonlinear verification)"
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
    p.add_argument("--group_by", default="utt_id",
                   help="Record field to group the train/val split by so the same "
                        "item can't leak across the split (utt_id, word, phone, "
                        "sentence, ...). Use 'none' for a plain random split.")
    p.add_argument("--records_path", type=Path, default=None,
                   help="Override path to the *_records.json used for the grouped split.")
    p.add_argument("--null_check", action="store_true",
                   help="Run one shuffled-target deflation per target on the first "
                        "Whisper model to confirm the chance floor of removed "
                        "held-out variance is ~0.")
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
    logger.info(f"  LLM target shape: {X_llm.shape}")

    N = len(X_kaldi)
    if args.group_by and args.group_by.lower() != "none":
        groups = load_group_ids(args.embeddings_dir, args.granularity,
                                args.group_by, N, args.records_path)
        val_mask = grouped_val_mask(groups, args.val_frac, args.seed)
        n_groups   = len(np.unique(groups))
        n_val_grp  = len(np.unique(groups[val_mask]))
        logger.info(f"Grouped split by '{args.group_by}': {n_groups:,} groups "
                    f"({n_val_grp:,} in val) — no item spans train/val")
    else:
        rng = np.random.default_rng(args.seed)
        val_mask = rng.random(N) < args.val_frac
        logger.warning("Ungrouped random split — train/val leakage possible if rows repeat")
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
    null_checked = False

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

        if args.null_check and not null_checked:
            logger.info("  [Null check] shuffled-target held-out removed variance (should be ~0)...")
            nf_k, nfr2_k = null_floor(X_w, X_kaldi, val_mask, args.seed)
            logger.info(f"    Kaldi(shuffled): removed_var={nf_k:.4f}  held-out R²={nfr2_k:.4f}")
            nf_l, nfr2_l = null_floor(X_w, X_llm, val_mask, args.seed)
            logger.info(f"    LLM(shuffled):   removed_var={nf_l:.4f}  held-out R²={nfr2_l:.4f}")
            null_checked = True

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
        "method":       "forward_ridge_whisper_on_target_heldout_plus_mlp",
        "group_by":     args.group_by,
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
