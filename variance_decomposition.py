#!/usr/bin/env python3
"""
variance_decomposition.py

Tests whether Whisper's word-level representations can be reduced to a
linear combination of acoustic (Kaldi) and textual (LLM) information, or
whether a significant fraction requires the joint product of both modalities.

Approach: cross-covariance SVD projection
-----------------------------------------
Find subspaces *within Whisper's own space* that align with:

  1. Kaldi embeddings                         → acoustic subspace
  2. LLM embeddings (beyond acoustic)         → textual subspace
  3. Joint Kaldi×LLM signal (beyond 1 and 2) → interaction subspace
  4. Everything else                          → residual

Each component is found by SVD of the cross-covariance between the
*residualised* Whisper embeddings and the reference signal, ensuring
orthogonality. Variance fractions = ||X projected onto subspace||² / ||X||²,
summing to exactly 1.

No regression or R² — this directly partitions Whisper's own variance.

Prerequisites
-------------
Run word_level_analysis.py first to generate:
    WordData/word_embeddings_{model_name}.pkl

Usage
-----
    python variance_decomposition.py
    python variance_decomposition.py --acoustic kaldi-librispeech --textual olmo-7b
    python variance_decomposition.py --k 100
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_DIR    = Path("WordData")
RESULTS_DIR = Path("DecompositionData")
PLOTS_DIR   = Path("DecompositionPlots")

DEFAULT_ACOUSTIC = "kaldi-librispeech"
DEFAULT_TEXTUAL  = "olmo-7b"

WHISPER_ENC_MODELS = [
    "whisper-base-enc", "whisper-small-enc", "whisper-medium-enc",
    "whisper-large-v1-enc", "whisper-large-v2-enc", "whisper-large-enc",
]
WHISPER_DEC_MODELS = [
    "whisper-base-dec", "whisper-small-dec", "whisper-medium-dec",
    "whisper-large-v1-dec", "whisper-large-v2-dec", "whisper-large-dec",
]

K_COMPONENTS = 50   # subspace dimensions to retain per component
K_INTERACT_A = 20   # PCA dims for Kaldi before building interaction features
K_INTERACT_T = 20   # PCA dims for LLM before building interaction features
MAX_SAMPLES  = 50_000
SEED         = 42

N_BOOTSTRAP  = 0        # 0 = skip; set to e.g. 200 to enable
BOOT_SIZE    = 10_000   # tokens per bootstrap resample (smaller → faster)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_embedding(model_name: str, data_dir: Path) -> np.ndarray | None:
    path = data_dir / f"word_embeddings_{model_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        emb = pickle.load(f)
    log.info(f"  Loaded {model_name}: {emb.shape}")
    return emb.astype(np.float32)


def list_available(data_dir: Path) -> list[str]:
    return sorted(
        p.stem.replace("word_embeddings_", "")
        for p in data_dir.glob("word_embeddings_*.pkl")
    )


def valid_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.linalg.norm(arr, axis=1) > 1e-10
    return mask


# ---------------------------------------------------------------------------
# Core projection utilities
# ---------------------------------------------------------------------------

def cross_cov_directions(A: np.ndarray, B: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the top-k directions in A's feature space that covary most with B.

    Computes randomized SVD of A^T @ B (the cross-covariance matrix) and
    returns the left singular vectors U ∈ R^{D_a × k} — directions in A's
    own space — and singular values σ (strength of cross-covariance).

    Parameters
    ----------
    A : (N, D_a)  embedding matrix whose subspace we are decomposing
    B : (N, D_b)  reference embedding matrix

    Returns
    -------
    U : (D_a, k)  orthonormal directions in A's space most aligned with B
    s : (k,)      corresponding singular values
    """
    C = A.T @ B                                              # (D_a, D_b)
    k = min(k, C.shape[0] - 1, C.shape[1] - 1)
    U, s, _ = randomized_svd(C, n_components=k, random_state=SEED)
    return U, s


def project_out(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Remove the subspace spanned by U (D × k, orthonormal cols) from X (N × D).
    Returns X with the U-subspace zeroed out.
    """
    return X - (X @ U) @ U.T


def variance_fraction(X: np.ndarray, U: np.ndarray) -> float:
    """
    Fraction of X's total variance lying in the subspace spanned by U.

    Since U has orthonormal columns:
        ||X U U^T||²_F  =  ||X U||²_F
    """
    return float(np.sum((X @ U) ** 2) / np.sum(X ** 2))


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def _orth_basis(*Us: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the union of column spaces of Us, via thin SVD."""
    Q, s, _ = np.linalg.svd(np.hstack(Us), full_matrices=False)
    return Q[:, s > 1e-10 * s[0]]


def _decompose_core(
    X: np.ndarray, K: np.ndarray, T: np.ndarray,
    k: int, k_int_a: int, k_int_t: int, seed: int,
) -> dict:
    """
    Inner decomposition on already-subsampled data. No logging, no subsampling.
    Assumes X, K, T are already centered and float64.
    """
    U_K, sv_K = cross_cov_directions(X, K, k)
    frac_acoustic = variance_fraction(X, U_K)

    X_perp_K = project_out(X, U_K)
    U_T, sv_T = cross_cov_directions(X_perp_K, T, min(k, T.shape[1] - 1))
    frac_textual = variance_fraction(X, U_T)

    X_perp_KT = project_out(X_perp_K, U_T)
    k_a = min(k_int_a, K.shape[1])
    k_t = min(k_int_t, T.shape[1])
    K_r = PCA(n_components=k_a, random_state=seed).fit_transform(K)
    T_r = PCA(n_components=k_t, random_state=seed).fit_transform(T)
    AT  = (K_r[:, :, None] * T_r[:, None, :]).reshape(len(X_perp_KT), -1)
    U_AT, sv_AT = cross_cov_directions(X_perp_KT, AT, min(k, AT.shape[1] - 1))
    frac_interaction = variance_fraction(X, U_AT)

    frac_residual = max(0.0, 1.0 - frac_acoustic - frac_textual - frac_interaction)
    return {
        "acoustic":       frac_acoustic,
        "textual":        frac_textual,
        "interaction":    frac_interaction,
        "residual":       frac_residual,
        "sv_acoustic":    sv_K.tolist(),
        "sv_textual":     sv_T.tolist(),
        "sv_interaction": sv_AT.tolist(),
    }


def shapley_decompose(
    X: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    k: int       = K_COMPONENTS,
    k_int_a: int = K_INTERACT_A,
    k_int_t: int = K_INTERACT_T,
    seed: int    = SEED,
) -> dict:
    """
    Shapley-value variance decomposition for acoustic (A), textual (T),
    and interaction (AT) components.

    Direction matrices are found WITHOUT sequential projection, so the
    subspaces for A, T, and AT may overlap. Coalition values v(S) use the
    orthonormal basis of the union of their subspaces, avoiding double-
    counting within each coalition.

    Returns:
      - Shapley value for each component (average marginal contribution
        across all 6 orderings — symmetric, order-invariant allocation)
      - interaction_upper_bound: v({AT}), the raw interaction variance with
        no projection — an upper bound on the interaction fraction
      - interaction_type3: v({A,T,AT}) - v({A,T}), the unique contribution
        of AT after both main effects are already in the model (Type III SS)

    Note: phi_A + phi_T + phi_AT = v({A,T,AT}) by Shapley efficiency.
    Whatever extra phi_AT gains vs. the sequential lower bound is taken
    directly from phi_A and phi_T, not created from nothing.
    """
    k_a = min(k_int_a, K.shape[1])
    k_t = min(k_int_t, T.shape[1])
    K_r = PCA(n_components=k_a, random_state=seed).fit_transform(K)
    T_r = PCA(n_components=k_t, random_state=seed).fit_transform(T)
    AT  = (K_r[:, :, None] * T_r[:, None, :]).reshape(len(X), -1)

    # Raw direction matrices — no sequential projection
    U_A,  _ = cross_cov_directions(X, K,  k)
    U_T,  _ = cross_cov_directions(X, T,  min(k, T.shape[1] - 1))
    U_AT, _ = cross_cov_directions(X, AT, min(k, AT.shape[1] - 1))

    # All 7 coalition variance fractions v(S) for S ⊆ {A, T, AT}
    vA    = variance_fraction(X, U_A)
    vT    = variance_fraction(X, U_T)
    vAT   = variance_fraction(X, U_AT)
    vA_T  = variance_fraction(X, _orth_basis(U_A, U_T))
    vA_AT = variance_fraction(X, _orth_basis(U_A, U_AT))
    vT_AT = variance_fraction(X, _orth_basis(U_T, U_AT))
    vFull = variance_fraction(X, _orth_basis(U_A, U_T, U_AT))

    # 3-player Shapley values (exact formula: average marginal contribution
    # over all orderings, weighted by |S|!(|N|-|S|-1)!/|N|!)
    phi_A  = vA/3 + (vA_T - vT)/6  + (vA_AT - vAT)/6 + (vFull - vT_AT)/3
    phi_T  = vT/3 + (vA_T - vA)/6  + (vT_AT - vAT)/6 + (vFull - vA_AT)/3
    phi_AT = vAT/3 + (vA_AT - vA)/6 + (vT_AT - vT)/6 + (vFull - vA_T)/3

    log.info(f"      Shapley acoustic:    {phi_A:.3f}")
    log.info(f"      Shapley textual:     {phi_T:.3f}")
    log.info(f"      Shapley interaction: {phi_AT:.3f}  "
             f"[lower={vFull - vA_T:.3f} (type3), upper={vAT:.3f}]")

    return {
        "shapley_acoustic":        float(phi_A),
        "shapley_textual":         float(phi_T),
        "shapley_interaction":     float(phi_AT),
        "shapley_residual":        float(max(0.0, 1.0 - phi_A - phi_T - phi_AT)),
        "interaction_upper_bound": float(vAT),
        "interaction_type3":       float(vFull - vA_T),
        "coalition_values": {
            "A": float(vA), "T": float(vT), "AT": float(vAT),
            "A_T": float(vA_T), "A_AT": float(vA_AT),
            "T_AT": float(vT_AT), "full": float(vFull),
        },
    }


def bootstrap_cis(
    X: np.ndarray, K: np.ndarray, T: np.ndarray,
    k: int       = K_COMPONENTS,
    k_int_a: int = K_INTERACT_A,
    k_int_t: int = K_INTERACT_T,
    n_bootstrap: int = N_BOOTSTRAP,
    boot_size: int   = BOOT_SIZE,
    seed: int        = SEED,
) -> dict:
    """
    Non-parametric bootstrap CIs for each variance fraction.

    Resamples `boot_size` tokens with replacement `n_bootstrap` times,
    re-fitting the full SVD decomposition each time. Returns per-fraction
    mean, std, and 95% CI (2.5th / 97.5th percentiles).
    """
    rng = np.random.default_rng(seed)
    N   = len(X)
    n   = min(boot_size, N)

    COMPS = ["acoustic", "textual", "interaction", "residual"]
    samples = {c: [] for c in COMPS}

    log.info(f"    Bootstrapping ({n_bootstrap} × {n:,} tokens) ...")
    for i in range(n_bootstrap):
        if (i + 1) % 20 == 0:
            log.info(f"      bootstrap {i+1}/{n_bootstrap}")
        idx = rng.choice(N, size=n, replace=True)
        Xb = (X[idx] - X[idx].mean(0)).astype(np.float64)
        Kb = (K[idx] - K[idx].mean(0)).astype(np.float64)
        Tb = (T[idx] - T[idx].mean(0)).astype(np.float64)
        res = _decompose_core(Xb, Kb, Tb, k, k_int_a, k_int_t, seed=seed + i)
        for c in COMPS:
            samples[c].append(res[c])

    cis = {}
    for comp, vals in samples.items():
        arr = np.array(vals)
        cis[comp] = {
            "mean":  float(arr.mean()),
            "std":   float(arr.std()),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
        }
        log.info(
            f"    {comp:12s}: {cis[comp]['mean']:.3f}  "
            f"95% CI [{cis[comp]['ci_lo']:.3f}, {cis[comp]['ci_hi']:.3f}]"
        )
    return cis


def decompose(
    X: np.ndarray,          # Whisper embeddings   (N, D_w)
    K: np.ndarray,          # Kaldi embeddings     (N, D_k)
    T: np.ndarray,          # LLM embeddings       (N, D_t)
    k: int       = K_COMPONENTS,
    k_int_a: int = K_INTERACT_A,
    k_int_t: int = K_INTERACT_T,
    max_samples: int = MAX_SAMPLES,
    seed: int    = SEED,
) -> dict:
    """
    Decompose X's variance into four orthogonal components.

    Steps
    -----
    1. Acoustic:     SVD of X^T K            → U_K  (directions in X ↔ Kaldi)
    2. Textual:      SVD of X_⊥K^T T        → U_T  (directions in X ↔ LLM, beyond acoustic)
    3. Interaction:  SVD of X_⊥KT^T (K⊗T)  → U_AT (directions in X ↔ joint signal, beyond both)
    4. Residual:     1 − (acoustic + textual + interaction)

    U_T ⊥ U_K and U_AT ⊥ U_K, U_T by construction (projecting out existing
    subspaces before each SVD step guarantees mutual orthogonality).
    """
    N = X.shape[0]

    # Subsample for speed
    rng = np.random.default_rng(seed)
    if N > max_samples:
        idx = rng.choice(N, size=max_samples, replace=False)
        X, K, T = X[idx], K[idx], T[idx]
        log.info(f"    Subsampled {N:,} → {max_samples:,} tokens")

    # Center and cast to float64
    X = (X - X.mean(axis=0)).astype(np.float64)
    K = (K - K.mean(axis=0)).astype(np.float64)
    T = (T - T.mean(axis=0)).astype(np.float64)

    log.info("    Finding acoustic subspace ...")
    log.info("    Finding textual subspace ...")
    log.info("    Finding interaction subspace ...")

    result = _decompose_core(X, K, T, k, k_int_a, k_int_t, seed)

    log.info(f"      Acoustic:    {result['acoustic']:.3f}  (top sv: {result['sv_acoustic'][0]:.2f})")
    log.info(f"      Textual:     {result['textual']:.3f}  (top sv: {result['sv_textual'][0]:.2f})")
    log.info(f"      Interaction: {result['interaction']:.3f}  (top sv: {result['sv_interaction'][0]:.2f})")
    log.info(f"      Residual:    {result['residual']:.3f}")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stacked(results: dict, acoustic_model: str, textual_model: str,
                 output_path: Path, cis: dict | None = None) -> None:
    size_order = ["base", "small", "medium", "large-v1", "large-v2", "large"]

    def sort_key(name):
        for i, s in enumerate(size_order):
            if s in name:
                return i
        return 99

    enc    = sorted([m for m in results if m.endswith("-enc")], key=sort_key)
    dec    = sorted([m for m in results if m.endswith("-dec")], key=sort_key)
    models = enc + dec

    if not models:
        log.warning("No results to plot.")
        return

    components = ["acoustic", "textual", "interaction", "residual"]
    colors     = ["#2196F3", "#4CAF50", "#FF9800", "#BDBDBD"]
    labels     = [
        f"Acoustic [{acoustic_model}]",
        f"Textual [{textual_model}]",
        "Interaction (joint; not a linear combination)",
        "Residual",
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.3), 6))
    x       = np.arange(len(models))
    bottoms = np.zeros(len(models))

    for comp, color, label in zip(components, colors, labels):
        vals = np.array([results[m].get(comp, 0.0) for m in models])
        bars = ax.bar(x, vals, bottom=bottoms, color=color,
                      label=label, width=0.65, edgecolor="white", linewidth=0.5)
        for bar, val, bot in zip(bars, vals, bottoms):
            if val > 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold",
                )
        # Error bars: CI on each segment's height, anchored at top of segment
        if cis is not None:
            for i, m in enumerate(models):
                if m not in cis or comp not in cis[m]:
                    continue
                ci = cis[m][comp]
                mid = bottoms[i] + vals[i]           # top of this segment
                lo  = vals[i] - ci["ci_lo"]          # distance to lower bound
                hi  = ci["ci_hi"] - vals[i]          # distance to upper bound
                ax.errorbar(
                    x[i], mid,
                    yerr=[[lo], [hi]],
                    fmt="none", color="black",
                    capsize=3, capthick=1, elinewidth=1,
                )
        bottoms += vals

    if enc and dec:
        ax.axvline(len(enc) - 0.5, color="black", linestyle="--", linewidth=1, alpha=0.4)
        ax.text(len(enc) / 2 - 0.5, 1.045, "Encoder",
                ha="center", fontsize=10, transform=ax.get_xaxis_transform())
        ax.text(len(enc) + len(dec) / 2 - 0.5, 1.045, "Decoder",
                ha="center", fontsize=10, transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("whisper-", "").replace("-enc", "").replace("-dec", "") for m in models],
        fontsize=10,
    )
    ax.set_ylabel("Fraction of Variance", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_title(
        "Whisper Embedding Variance Decomposition\n"
        f"Acoustic ref: {acoustic_model}  ·  Textual ref: {textual_model}",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved plot → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Whisper variance decomposition via cross-covariance SVD projection"
    )
    parser.add_argument("--data_dir",       type=Path, default=DATA_DIR)
    parser.add_argument("--results_dir",    type=Path, default=RESULTS_DIR)
    parser.add_argument("--plots_dir",      type=Path, default=PLOTS_DIR)
    parser.add_argument("--acoustic",       type=str,  default=DEFAULT_ACOUSTIC)
    parser.add_argument("--textual",        type=str,  default=DEFAULT_TEXTUAL)
    parser.add_argument("--k",              type=int,  default=K_COMPONENTS,
                        help="Subspace dimensions to retain per component")
    parser.add_argument("--k_interact_a",   type=int,  default=K_INTERACT_A,
                        help="PCA dims for Kaldi before building interaction features")
    parser.add_argument("--k_interact_t",   type=int,  default=K_INTERACT_T,
                        help="PCA dims for LLM before building interaction features")
    parser.add_argument("--max_samples",    type=int,  default=MAX_SAMPLES)
    parser.add_argument("--n_bootstrap",    type=int,  default=N_BOOTSTRAP,
                        help="Bootstrap resamples for CIs (0 = skip)")
    parser.add_argument("--boot_size",      type=int,  default=BOOT_SIZE,
                        help="Tokens per bootstrap resample")
    parser.add_argument("--shapley",        action="store_true",
                        help="Compute Shapley values, upper bound, and Type III interaction")
    parser.add_argument("--whisper_models", nargs="+",
                        default=WHISPER_ENC_MODELS + WHISPER_DEC_MODELS)
    args = parser.parse_args()

    args.results_dir.mkdir(exist_ok=True)
    args.plots_dir.mkdir(exist_ok=True)

    available = list_available(args.data_dir)
    if not available:
        log.error(
            f"No word_embeddings_*.pkl files found in {args.data_dir}.\n"
            "Re-run word_level_analysis.py to generate them, then run this script."
        )
        return
    log.info(f"Available embeddings: {available}")

    log.info(f"\nLoading acoustic reference: {args.acoustic}")
    K_emb = load_embedding(args.acoustic, args.data_dir)
    if K_emb is None:
        log.error(f"Acoustic reference '{args.acoustic}' not found. Available: {available}")
        return

    log.info(f"Loading textual reference: {args.textual}")
    T_emb = load_embedding(args.textual, args.data_dir)
    if T_emb is None:
        log.error(f"Textual reference '{args.textual}' not found. Available: {available}")
        return

    all_results  = {}
    all_cis      = {}
    all_shapley  = {}

    for model_name in args.whisper_models:
        log.info(f"\n{'=' * 60}")
        log.info(f"  {model_name}")
        log.info(f"{'=' * 60}")

        X = load_embedding(model_name, args.data_dir)
        if X is None:
            log.warning(f"  Skipping {model_name} — pkl not found.")
            continue

        mask = valid_mask(X, K_emb, T_emb)
        log.info(f"  Valid tokens: {mask.sum():,} / {len(mask):,}")

        result = decompose(
            X[mask], K_emb[mask], T_emb[mask],
            k=args.k,
            k_int_a=args.k_interact_a,
            k_int_t=args.k_interact_t,
            max_samples=args.max_samples,
        )
        all_results[model_name] = result

        log.info(f"  Acoustic:    {result['acoustic']:.3f}")
        log.info(f"  Textual:     {result['textual']:.3f}")
        log.info(f"  Interaction: {result['interaction']:.3f}")
        log.info(f"  Residual:    {result['residual']:.3f}")

        if args.shapley:
            log.info("  Computing Shapley decomposition ...")
            rng_sub = np.random.default_rng(SEED)
            Xm, Km, Tm = X[mask], K_emb[mask], T_emb[mask]
            N_valid = len(Xm)
            if N_valid > args.max_samples:
                idx = rng_sub.choice(N_valid, size=args.max_samples, replace=False)
                Xm, Km, Tm = Xm[idx], Km[idx], Tm[idx]
            Xm = (Xm - Xm.mean(0)).astype(np.float64)
            Km = (Km - Km.mean(0)).astype(np.float64)
            Tm = (Tm - Tm.mean(0)).astype(np.float64)
            all_shapley[model_name] = shapley_decompose(
                Xm, Km, Tm,
                k=args.k,
                k_int_a=args.k_interact_a,
                k_int_t=args.k_interact_t,
            )

        if args.n_bootstrap > 0:
            # Bootstrap on the same (possibly subsampled) token pool
            rng_sub = np.random.default_rng(SEED)
            Xm, Km, Tm = X[mask], K_emb[mask], T_emb[mask]
            N_valid = len(Xm)
            if N_valid > args.max_samples:
                idx = rng_sub.choice(N_valid, size=args.max_samples, replace=False)
                Xm, Km, Tm = Xm[idx], Km[idx], Tm[idx]
            all_cis[model_name] = bootstrap_cis(
                Xm, Km, Tm,
                k=args.k,
                k_int_a=args.k_interact_a,
                k_int_t=args.k_interact_t,
                n_bootstrap=args.n_bootstrap,
                boot_size=args.boot_size,
            )

    if not all_results:
        log.error("No results computed — check that pkl files exist.")
        return

    out_json = args.results_dir / "decomposition_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "acoustic_model": args.acoustic,
            "textual_model":  args.textual,
            "k":            args.k,
            "k_interact_a": args.k_interact_a,
            "k_interact_t": args.k_interact_t,
            "max_samples":  args.max_samples,
            "n_bootstrap":  args.n_bootstrap,
            "boot_size":    args.boot_size,
            "results":      all_results,
            "bootstrap_cis": all_cis,
            "shapley":      all_shapley,
        }, f, indent=2)
    log.info(f"\nSaved → {out_json}")

    out_csv = args.results_dir / "decomposition_summary.csv"
    with open(out_csv, "w") as f:
        has_ci      = bool(all_cis)
        has_shapley = bool(all_shapley)
        header = "model,acoustic,textual,interaction_lower,residual"
        if has_shapley:
            header += (",interaction_upper,interaction_type3"
                       ",shapley_acoustic,shapley_textual"
                       ",shapley_interaction,shapley_residual")
        if has_ci:
            header += (",interaction_ci_lo,interaction_ci_hi"
                       ",acoustic_ci_lo,acoustic_ci_hi"
                       ",textual_ci_lo,textual_ci_hi")
        f.write(header + "\n")
        for model_name, res in all_results.items():
            row = (f"{model_name},{res['acoustic']:.4f},{res['textual']:.4f},"
                   f"{res['interaction']:.4f},{res['residual']:.4f}")
            if has_shapley and model_name in all_shapley:
                sh = all_shapley[model_name]
                row += (f",{sh['interaction_upper_bound']:.4f}"
                        f",{sh['interaction_type3']:.4f}"
                        f",{sh['shapley_acoustic']:.4f}"
                        f",{sh['shapley_textual']:.4f}"
                        f",{sh['shapley_interaction']:.4f}"
                        f",{sh['shapley_residual']:.4f}")
            if has_ci and model_name in all_cis:
                ci = all_cis[model_name]
                row += (f",{ci['interaction']['ci_lo']:.4f},{ci['interaction']['ci_hi']:.4f}"
                        f",{ci['acoustic']['ci_lo']:.4f},{ci['acoustic']['ci_hi']:.4f}"
                        f",{ci['textual']['ci_lo']:.4f},{ci['textual']['ci_hi']:.4f}")
            f.write(row + "\n")
    log.info(f"Saved → {out_csv}")

    plot_path = args.plots_dir / f"decomposition_{args.acoustic}_vs_{args.textual}.png"
    plot_stacked(all_results, args.acoustic, args.textual, plot_path,
                 cis=all_cis if all_cis else None)


if __name__ == "__main__":
    main()
