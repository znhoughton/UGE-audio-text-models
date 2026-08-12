#!/usr/bin/env python3
"""
Whisper-Kaldi-LLM Variance Decomposition via Iterative MLP Deflation

Decomposes Whisper encoder embedding variance into four components:
  - Kaldi-only    : variance predictable from Kaldi but not LLMs
  - LLM-only      : variance predictable from LLMs but not Kaldi
  - Shared        : variance predictable from both
  - Neither       : variance predictable from neither

Method:
  For each Whisper model, run deflation in both orderings:

  Ordering 1 (Kaldi first):
    1. Train MLP f1: X_kaldi -> X_whisper_residual; subtract f1(X_kaldi); repeat until R² < threshold
    2. Train MLP f2: X_llm   -> residual;            subtract; repeat

  Ordering 2 (LLM first):
    1. Train MLP g1: X_llm   -> X_whisper_residual; subtract; repeat
    2. Train MLP g2: X_kaldi -> residual;            subtract; repeat

  Four components recovered:
    kaldi_only = V_kaldi_given_llm  (from ordering 2, step 2)
    llm_only   = V_llm_given_kaldi  (from ordering 1, step 2)
    shared     = V_kaldi_total - kaldi_only  (= V_llm_total - llm_only, sanity check)
    neither    = 1 - kaldi_only - llm_only - shared

Usage (word-level):
  python projection_analysis.py --granularity word \
      --embeddings_dir /dpluth-data \
      --output_dir ./word

Usage (phone-level):
  python projection_analysis.py --granularity phone \
      --embeddings_dir /dpluth-data \
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


# ---------------------------------------------------------------------------
# MLP probe
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

    All data lives GPU-resident for the duration of training to avoid
    repeated CPU→GPU transfers. Shuffling is done with torch.randperm on
    the GPU rather than through DataLoader (which is optimised for CPU data).
    Mixed precision (bfloat16) is used on CUDA for ~2x throughput.
    """
    probe = MLPProbe(X_in_tr.shape[1], X_out_tr.shape[1],
                     hidden_dim, n_layers, dropout).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Move full splits to GPU once
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


# ---------------------------------------------------------------------------
# Iterative deflation
# ---------------------------------------------------------------------------

def iterative_deflation(
    X_whisper_residual: np.ndarray,
    X_target: np.ndarray,
    val_mask: np.ndarray,
    total_var: float,
    threshold: float = 0.01,
    max_iters: int = 20,
    device: str = "cuda",
    **probe_kwargs,
) -> tuple:
    """
    Repeatedly train MLP (X_target -> residual), subtract predictions, until R² < threshold.

    Probe direction: X_target -> X_whisper_residual
      - f(X_target) is in Whisper space so subtraction is well-defined.
      - Dimensions of Whisper not predictable from X_target get near-zero predictions
        (conditional mean ≈ 0 when embeddings are centered) and survive intact.

    Returns:
        residual (np.ndarray): deflated Whisper embeddings
        log (list[dict]):      per-iteration R² and variance removed
    """
    train_mask = ~val_mask
    residual = X_whisper_residual.copy()
    log = []

    for i in range(max_iters):
        probe = train_probe(
            X_target[train_mask], residual[train_mask],
            X_target[val_mask],   residual[val_mask],
            device=device, **probe_kwargs,
        )

        pred_val = predict_batched(probe, X_target[val_mask], device=device)
        r2 = r2_score(residual[val_mask], pred_val)

        pred_all = predict_batched(probe, X_target, device=device)
        var_removed = float((pred_all ** 2).sum() / total_var)

        entry = {"iteration": i + 1, "r2_val": round(r2, 5),
                 "var_removed_frac": round(var_removed, 5)}
        log.append(entry)
        logger.info(f"    iter {i + 1}: R²={r2:.4f}  var_removed={var_removed:.4f}")

        residual = residual - pred_all

        del probe, pred_all, pred_val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if r2 < threshold:
            logger.info(f"    R² < {threshold} — deflation complete")
            break

    return residual, log


# ---------------------------------------------------------------------------
# Venn decomposition
# ---------------------------------------------------------------------------

def venn_decomposition(
    X_whisper: np.ndarray,
    X_kaldi: np.ndarray,
    X_llm: np.ndarray,
    val_mask: np.ndarray,
    threshold: float = 0.01,
    max_iters: int = 20,
    device: str = "cuda",
    **probe_kwargs,
) -> dict:
    """
    Run both orderings of deflation and recover four variance components.

    From two orderings:
      Ordering 1 — Kaldi first:
        V_kaldi_total   = Kaldi-only + Shared
        V_llm_given_k   = LLM-only
      Ordering 2 — LLM first:
        V_llm_total     = LLM-only + Shared
        V_kaldi_given_l = Kaldi-only

    Therefore:
      kaldi_only = V_kaldi_given_l
      llm_only   = V_llm_given_k
      shared     = V_kaldi_total - kaldi_only  (== V_llm_total - llm_only)
      neither    = 1 - kaldi_only - llm_only - shared
    """
    total_var = float((X_whisper ** 2).sum())

    def vfrac(arr):
        return float((arr ** 2).sum() / total_var)

    deflation_kw = dict(val_mask=val_mask, total_var=total_var,
                        threshold=threshold, max_iters=max_iters,
                        device=device, **probe_kwargs)

    # ---- Ordering 1: Kaldi → LLM ----------------------------------------
    logger.info("  [Ordering 1] Deflating Kaldi...")
    res_k, log_k = iterative_deflation(X_whisper, X_kaldi, **deflation_kw)

    logger.info("  [Ordering 1] Deflating LLM from Kaldi residual...")
    res_kl, log_kl = iterative_deflation(res_k, X_llm, **deflation_kw)

    V_kaldi_total  = 1.0 - vfrac(res_k)
    V_llm_given_k  = vfrac(res_k) - vfrac(res_kl)
    V_neither_ord1 = vfrac(res_kl)

    # ---- Ordering 2: LLM → Kaldi ----------------------------------------
    logger.info("  [Ordering 2] Deflating LLM...")
    res_l, log_l = iterative_deflation(X_whisper, X_llm, **deflation_kw)

    logger.info("  [Ordering 2] Deflating Kaldi from LLM residual...")
    res_lk, log_lk = iterative_deflation(res_l, X_kaldi, **deflation_kw)

    V_llm_total    = 1.0 - vfrac(res_l)
    V_kaldi_given_l = vfrac(res_l) - vfrac(res_lk)
    V_neither_ord2 = vfrac(res_lk)

    # ---- Four components -------------------------------------------------
    kaldi_only = V_kaldi_given_l
    llm_only   = V_llm_given_k
    shared     = V_kaldi_total - kaldi_only
    neither    = (V_neither_ord1 + V_neither_ord2) / 2.0

    shared_check = V_llm_total - llm_only
    if abs(shared - shared_check) > 0.02:
        logger.warning(f"  Shared inconsistency: {shared:.4f} vs {shared_check:.4f} — "
                       "may indicate probe didn't converge fully")
    if abs(V_neither_ord1 - V_neither_ord2) > 0.02:
        logger.warning(f"  Neither inconsistency: ord1={V_neither_ord1:.4f} ord2={V_neither_ord2:.4f}")

    return {
        "kaldi_only":    round(kaldi_only, 5),
        "llm_only":      round(llm_only, 5),
        "shared":        round(shared, 5),
        "neither":       round(neither, 5),
        # Raw quantities for transparency / debugging
        "V_kaldi_total":    round(V_kaldi_total, 5),
        "V_llm_total":      round(V_llm_total, 5),
        "V_kaldi_given_llm": round(V_kaldi_given_l, 5),
        "V_llm_given_kaldi": round(V_llm_given_k, 5),
        "neither_ordering1": round(V_neither_ord1, 5),
        "neither_ordering2": round(V_neither_ord2, 5),
        "shared_check":      round(shared_check, 5),
        "logs": {
            "ordering1_kaldi": log_k,
            "ordering1_llm":   log_kl,
            "ordering2_llm":   log_l,
            "ordering2_kaldi": log_lk,
        },
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embedding(embeddings_dir: Path, granularity: str, model_name: str) -> np.ndarray:
    subdir  = "WordData"  if granularity == "word"  else "PhoneData"
    prefix  = "word"      if granularity == "word"  else "phone"
    path = embeddings_dir / subdir / f"{prefix}_embeddings_{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Embedding not found: {path}")
    logger.info(f"  Loading {path}")
    with open(path, "rb") as f:
        emb = pickle.load(f)
    return emb.astype(np.float32)


def build_llm_target(embeddings_dir: Path, granularity: str,
                     llm_models: list, pca_dims: int, seed: int) -> np.ndarray:
    """
    If one LLM: return its embeddings directly.
    If multiple LLMs: compute a shared subspace via PCA on pooled (centered,
    scale-normalized) embeddings, return projection onto top pca_dims PCs.
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
            # Normalize each model to unit average norm before pooling so no
            # single high-dimensional model dominates the shared subspace
            e /= (np.linalg.norm(e, axis=1).mean() + 1e-8)
            arrays.append(e)
        except FileNotFoundError as ex:
            logger.warning(f"  {ex} — skipping")

    if not arrays:
        raise RuntimeError("No LLM embeddings found")

    stacked = np.concatenate(arrays, axis=1)   # (N, sum_d)
    del arrays; gc.collect()

    stacked -= stacked.mean(0)
    stacked /= np.sqrt(stacked.shape[0] - 1)   # scale for SVD

    from sklearn.utils.extmath import randomized_svd
    _, _, Vt = randomized_svd(stacked, n_components=pca_dims, random_state=seed)
    del stacked; gc.collect()

    # Project: reload and re-concatenate at float32 for the final projection
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

    projection = stacked2 @ Vt.T.astype(np.float32)   # (N, pca_dims)
    del stacked2; gc.collect()
    logger.info(f"  Shared LLM target shape: {projection.shape}")
    return projection


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Whisper-Kaldi-LLM variance decomposition"
    )
    p.add_argument("--granularity", choices=["word", "phone"], required=True,
                   help="Analysis granularity")
    p.add_argument("--embeddings_dir", type=Path, default=Path("/dpluth-data"),
                   help="Root dir containing WordData/ and PhoneData/ pkl files")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where to save results (default: ./<granularity>/)")
    p.add_argument("--whisper_models", nargs="+", default=WHISPER_ALL,
                   help="Whisper model(s) to analyse (encoders + decoders by default)")
    p.add_argument("--kaldi_model", default="kaldi-librispeech",
                   choices=KALDI_MODELS)
    p.add_argument("--llm_models", nargs="+", default=["olmo-7b"],
                   help="LLM(s) to use. Defaults to olmo-7b (largest). Multiple → shared PCA subspace.")
    p.add_argument("--llm_pca_dims", type=int, default=256,
                   help="PCA dims for shared LLM subspace when >1 LLM given")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="R² threshold to stop iterative deflation")
    p.add_argument("--max_iters", type=int, default=20,
                   help="Max deflation iterations per stage")
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction of data held out for R² evaluation")
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
            threshold=args.threshold, max_iters=args.max_iters,
            device=device, **probe_kwargs,
        )

        logger.info(f"  --- {whisper_name} summary ---")
        logger.info(f"  Kaldi-only : {result['kaldi_only']:.4f}")
        logger.info(f"  LLM-only   : {result['llm_only']:.4f}")
        logger.info(f"  Shared     : {result['shared']:.4f}")
        logger.info(f"  Neither    : {result['neither']:.4f}")
        logger.info(f"  (check: sum = {result['kaldi_only']+result['llm_only']+result['shared']+result['neither']:.4f})")

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
        "threshold":    args.threshold,
        "max_iters":    args.max_iters,
        "results":      all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results JSON → {out_path}")

    # ------------------------------------------------------------------
    # Save per-model CSVs + summary CSV
    # ------------------------------------------------------------------
    COMPONENTS = ["kaldi_only", "llm_only", "shared", "neither",
                  "V_kaldi_total", "V_llm_total"]

    for whisper_name, result in all_results.items():
        csv_path = output_dir / f"{whisper_name}_decomposition.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["component", "fraction"])
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
