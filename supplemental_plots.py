#!/usr/bin/env python3
"""
Supplemental plots for representation_analysis.py results.

Reads from the same Data/ and Plots/ directories produced by the main script.
Does NOT require GPU or model loading — works entirely from cached .pkl embeddings
and results.json.

New plots produced:
  Plots/whisper_enc_variance_by_pc.png   — fraction of variance per PC, all Whisper encoders
  Plots/whisper_dec_variance_by_pc.png   — fraction of variance per PC, all Whisper decoders
  Plots/family_opt_variance_by_pc.png    — OPT family (babylm-125m, opt-125m, babylm-350m, babylm-1.3b)
  Plots/family_pythia_variance_by_pc.png — Pythia family (pythia-160m, pythia-6.9b)
  Plots/family_whisper_variance_by_pc.png— Whisper enc+dec together (all sizes)
  Plots/cumvar_all_models.png            — cumulative variance curves, all models overlaid
  Plots/participation_ratio.png          — participation ratio (alt effective rank) bar chart
  Plots/cka_within_family.png            — within-family CKA heatmaps side by side
  Plots/cka_audio_vs_text_strip.png      — strip plot: audio model CKA with each text LLM
  Plots/encoder_vs_decoder_cka.png       — enc vs dec CKA for each Whisper size
  Plots/embedding_dim_vs_effective_rank.png — scatter: model size vs effective rank

Usage:
    python supplemental_plots.py                        # default: root = .
    python supplemental_plots.py --root_dir /path/to/run
    python supplemental_plots.py --max_pc 20            # show first 20 PCs (default 15)
    python supplemental_plots.py --no_embeddings        # skip plots that need .pkl files;
                                                        # use results.json eigenvalues only
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Model metadata (mirrors main script)
# ---------------------------------------------------------------------------

MODEL_META = {
    "whisper-base-enc":   {"family": "whisper-enc",  "size_m": 74,    "label": "base",   "modality": "audio"},
    "whisper-base-dec":   {"family": "whisper-dec",  "size_m": 74,    "label": "base",   "modality": "audio"},
    "whisper-small-enc":  {"family": "whisper-enc",  "size_m": 244,   "label": "small",  "modality": "audio"},
    "whisper-small-dec":  {"family": "whisper-dec",  "size_m": 244,   "label": "small",  "modality": "audio"},
    "whisper-medium-enc": {"family": "whisper-enc",  "size_m": 769,   "label": "medium", "modality": "audio"},
    "whisper-medium-dec": {"family": "whisper-dec",  "size_m": 769,   "label": "medium", "modality": "audio"},
    "whisper-large-enc":  {"family": "whisper-enc",  "size_m": 1550,  "label": "large",  "modality": "audio"},
    "whisper-large-dec":  {"family": "whisper-dec",  "size_m": 1550,  "label": "large",  "modality": "audio"},
    "parakeet-ctc-0.6b":  {"family": "parakeet",     "size_m": 600,   "label": "0.6b",   "modality": "audio"},
    "mimi":               {"family": "mimi",          "size_m": 85,    "label": "mimi",   "modality": "audio"},
    "voxtral-3b":         {"family": "voxtral",       "size_m": 3000,  "label": "3b",     "modality": "audio"},
    "higgs-audio-v2-3b":  {"family": "higgs",         "size_m": 5800,  "label": "3b",     "modality": "tts"},
    "babylm-125m":        {"family": "opt",            "size_m": 125,   "label": "125m\nBabyLM", "modality": "text"},
    "opt-125m":           {"family": "opt",            "size_m": 125,   "label": "125m\nOPT",    "modality": "text"},
    "babylm-350m":        {"family": "opt",            "size_m": 350,   "label": "350m\nBabyLM", "modality": "text"},
    "babylm-1.3b":        {"family": "opt",            "size_m": 1300,  "label": "1.3b\nBabyLM", "modality": "text"},
    "olmo-7b":            {"family": "olmo",           "size_m": 7000,  "label": "7b",     "modality": "text"},
    "pythia-160m":        {"family": "pythia",         "size_m": 160,   "label": "160m",   "modality": "text"},
    "pythia-6.9b":        {"family": "pythia",         "size_m": 6900,  "label": "6.9b",   "modality": "text"},
}

# Colour palette — consistent with main script
MODEL_COLORS = {
    "whisper-base-enc":   "#BBDEFB",
    "whisper-base-dec":   "#90CAF9",
    "whisper-small-enc":  "#64B5F6",
    "whisper-small-dec":  "#42A5F5",
    "whisper-medium-enc": "#1E88E5",
    "whisper-medium-dec": "#1565C0",
    "whisper-large-enc":  "#0D47A1",
    "whisper-large-dec":  "#283593",
    "parakeet-ctc-0.6b":  "#00838F",
    "mimi":               "#D84315",
    "voxtral-3b":         "#FF6F00",
    "higgs-audio-v2-3b":  "#558B2F",
    "babylm-125m":        "#E65100",
    "opt-125m":           "#FFCCBC",
    "babylm-350m":        "#FB8C00",
    "babylm-1.3b":        "#F9A825",
    "olmo-7b":            "#2E7D32",
    "pythia-160m":        "#CE93D8",
    "pythia-6.9b":        "#6A1B9A",
}

FAMILY_PALETTES = {
    "whisper-enc": ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"],
    "whisper-dec": ["#90CAF9", "#42A5F5", "#1565C0", "#283593"],
    "opt":         ["#E65100", "#FFCCBC", "#FB8C00", "#F9A825"],
    "pythia":      ["#CE93D8", "#6A1B9A"],
    "olmo":        ["#2E7D32"],
}

AUDIO_MODALITIES = {"audio", "tts"}

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.color":       "#DDDDDD",
    "grid.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "DejaVu Sans",
}


def _style():
    plt.rcParams.update(PLOT_STYLE)


def effective_rank(eigs: np.ndarray) -> float:
    p = np.array(eigs, dtype=float)
    p = p / p.sum()
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio(eigs: np.ndarray) -> float:
    """Alternative dimensionality measure: PR = (Σλ)² / Σλ²"""
    lam = np.array(eigs, dtype=float)
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if len(lam) else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(data_dir: Path) -> dict:
    p = data_dir / "results.json"
    if not p.exists():
        sys.exit(f"[ERROR] results.json not found at {p}.\n"
                 f"Run the main script first, or pass --root_dir correctly.")
    with open(p) as f:
        return json.load(f)


def load_eigenvalues(results: dict) -> dict[str, np.ndarray]:
    """Load eigenvalue spectra from results.json."""
    return {k: np.array(v) for k, v in results.get("eigenvalue_spectra", {}).items()}


def load_embeddings_if_present(data_dir: Path, names: list) -> dict[str, np.ndarray]:
    """Silently skip models whose .pkl files are absent."""
    out = {}
    for name in names:
        p = data_dir / f"embeddings_{name}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                out[name] = pickle.load(f)
            print(f"  Loaded embeddings: {name}  {out[name].shape}")
        else:
            print(f"  [skip] No embedding cache for {name}")
    return out


# ---------------------------------------------------------------------------
# Shared helper: fraction-of-variance line plot for a family
# ---------------------------------------------------------------------------

def _variance_by_pc_plot(
    ax,
    names: list,
    eigenvalues: dict,
    max_pc: int,
    title: str,
    show_legend: bool = True,
):
    """Plot fraction-of-variance curves for `names` onto `ax`."""
    for name in names:
        if name not in eigenvalues:
            continue
        eigs = eigenvalues[name][:max_pc]
        k = np.arange(1, len(eigs) + 1)
        color = MODEL_COLORS.get(name, "#555")
        label = MODEL_META.get(name, {}).get("label", name)
        ax.plot(k, eigs, color=color, linewidth=2.2, label=label, alpha=0.9)
        ax.fill_between(k, eigs, alpha=0.08, color=color)

    ax.set_xlabel(f"Principal Component (1–{max_pc})", fontsize=10)
    ax.set_ylabel("Fraction of Variance", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(1, max_pc)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    if show_legend:
        ax.legend(fontsize=9, framealpha=0.9,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)


# ---------------------------------------------------------------------------
# Plot 1 & 2: Whisper encoder / decoder variance by PC
# ---------------------------------------------------------------------------

def plot_whisper_enc_variance(eigenvalues: dict, plots_dir: Path, max_pc: int):
    _style()
    enc_names = [n for n in MODEL_META if MODEL_META[n]["family"] == "whisper-enc" and n in eigenvalues]
    enc_names.sort(key=lambda n: MODEL_META[n]["size_m"])

    fig, ax = plt.subplots(figsize=(9, 5))
    _variance_by_pc_plot(ax, enc_names, eigenvalues, max_pc,
                         title="Whisper Encoder — Fraction of Variance by PC\n(all model sizes overlaid)")
    fig.suptitle("", fontsize=1)  # suppress default suptitle spacing
    plt.tight_layout()
    p = plots_dir / "whisper_enc_variance_by_pc.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


def plot_whisper_dec_variance(eigenvalues: dict, plots_dir: Path, max_pc: int):
    _style()
    dec_names = [n for n in MODEL_META if MODEL_META[n]["family"] == "whisper-dec" and n in eigenvalues]
    dec_names.sort(key=lambda n: MODEL_META[n]["size_m"])

    fig, ax = plt.subplots(figsize=(9, 5))
    _variance_by_pc_plot(ax, dec_names, eigenvalues, max_pc,
                         title="Whisper Decoder — Fraction of Variance by PC\n(all model sizes overlaid)")
    plt.tight_layout()
    p = plots_dir / "whisper_dec_variance_by_pc.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


# ---------------------------------------------------------------------------
# Plot 3–5: Per-family variance by PC
# ---------------------------------------------------------------------------

FAMILY_PLOT_CONFIGS = {
    "opt": {
        "title": "OPT Family — Fraction of Variance by PC\n(BabyLM sizes + full OPT-125m)",
        "filename": "family_opt_variance_by_pc.png",
    },
    "pythia": {
        "title": "Pythia Family — Fraction of Variance by PC\n(160m vs 6.9b, same corpus)",
        "filename": "family_pythia_variance_by_pc.png",
    },
    "whisper-enc": {
        "title": "Whisper Encoder Family — Fraction of Variance by PC",
        "filename": "family_whisper_enc_variance_by_pc.png",
    },
    "whisper-dec": {
        "title": "Whisper Decoder Family — Fraction of Variance by PC",
        "filename": "family_whisper_dec_variance_by_pc.png",
    },
    "olmo": {
        "title": "OLMo-2 7B — Fraction of Variance by PC",
        "filename": "family_olmo_variance_by_pc.png",
    },
}


def plot_family_variance(eigenvalues: dict, plots_dir: Path, max_pc: int):
    """One plot per family defined in FAMILY_PLOT_CONFIGS."""
    for family, cfg in FAMILY_PLOT_CONFIGS.items():
        names = [n for n in MODEL_META
                 if MODEL_META[n]["family"] == family and n in eigenvalues]
        names.sort(key=lambda n: MODEL_META[n]["size_m"])
        if not names:
            print(f"  [skip] No eigenvalues for family '{family}'")
            continue
        _style()
        fig, ax = plt.subplots(figsize=(9, 5))
        _variance_by_pc_plot(ax, names, eigenvalues, max_pc, title=cfg["title"])
        plt.tight_layout()
        p = plots_dir / cfg["filename"]
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {p}")


# ---------------------------------------------------------------------------
# Plot 6: Cumulative variance — all models overlaid
# ---------------------------------------------------------------------------

def plot_cumulative_variance(eigenvalues: dict, plots_dir: Path, max_pc: int):
    _style()
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, eigs in eigenvalues.items():
        eigs_k = eigs[:max_pc]
        cumvar = np.cumsum(eigs_k)
        k = np.arange(1, len(cumvar) + 1)
        color = MODEL_COLORS.get(name, "#555")
        meta = MODEL_META.get(name, {})
        modality = meta.get("modality", "text")
        ls = "--" if modality in AUDIO_MODALITIES else "-"
        ax.plot(k, cumvar, color=color, linewidth=2, linestyle=ls,
                label=name, alpha=0.85)

    ax.axhline(0.90, color="#888", linewidth=1, linestyle=":", alpha=0.6)
    ax.text(max_pc * 0.98, 0.915, "90% var", ha="right", fontsize=8, color="#888")
    ax.axhline(0.95, color="#888", linewidth=1, linestyle=":", alpha=0.4)
    ax.text(max_pc * 0.98, 0.965, "95% var", ha="right", fontsize=8, color="#888")

    ax.set_xlim(1, max_pc)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Number of Principal Components (1–{max_pc})", fontsize=11)
    ax.set_ylabel("Cumulative Fraction of Variance", fontsize=11)
    ax.set_title(
        "Cumulative Variance — All Models\n"
        "Solid = text LLMs  ·  Dashed = audio/speech models",
        fontsize=12, fontweight="bold",
    )

    # Legend: placed below the plot to never overlap lines
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, ncol=4, framealpha=0.9,
               loc="upper center", bbox_to_anchor=(0.5, -0.02),
               title="Solid = text LLMs  ·  Dashed = audio/speech")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    p = plots_dir / "cumvar_all_models.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


# ---------------------------------------------------------------------------
# Plot 7: Participation ratio bar chart
# ---------------------------------------------------------------------------

def plot_participation_ratio(eigenvalues: dict, plots_dir: Path):
    _style()
    names = list(eigenvalues.keys())
    prs = [participation_ratio(eigenvalues[n]) for n in names]
    ers = [effective_rank(eigenvalues[n]) for n in names]
    colors = [MODEL_COLORS.get(n, "#555") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(names) * 1.3), 4.5))

    for ax, values, ylabel, title_suffix in zip(
        axes,
        [prs, ers],
        ["Participation Ratio  (Σλ)² / Σλ²", "Effective Rank  exp(H(p))"],
        ["Participation Ratio", "Effective Rank"],
    ):
        bars = ax.bar(names, values, color=colors, width=0.55,
                      edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title_suffix, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, rotation=25)

    fig.suptitle(
        "Intrinsic Dimensionality Measures\n"
        "Both capture how concentrated variance is — higher = more spread out",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    p = plots_dir / "participation_ratio.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


# ---------------------------------------------------------------------------
# Plot 8: Within-family CKA heatmaps
# ---------------------------------------------------------------------------

def plot_within_family_cka(results: dict, plots_dir: Path):
    """Side-by-side mini heatmaps for each family that has ≥2 members."""
    cka_matrix = np.array(results.get("cka_matrix", []))
    names = list(results.get("embedding_shapes", {}).keys())
    if cka_matrix.size == 0 or not names:
        print("  [skip] No CKA matrix in results.json")
        return

    families = {}
    for n in names:
        fam = MODEL_META.get(n, {}).get("family", "other")
        families.setdefault(fam, []).append(n)

    multi_fam = {f: ms for f, ms in families.items() if len(ms) >= 2}
    if not multi_fam:
        print("  [skip] No families with ≥2 models for within-family CKA")
        return

    n_fam = len(multi_fam)
    fig, axes = plt.subplots(1, n_fam, figsize=(4.5 * n_fam, 4.5))
    if n_fam == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list(
        "cka", ["#FFFFFF", "#A5D6A7", "#2E7D32"], N=256
    )

    for ax, (fam, members) in zip(axes, multi_fam.items()):
        idxs = [names.index(m) for m in members if m in names]
        members = [names[i] for i in idxs]
        sub = cka_matrix[np.ix_(idxs, idxs)]
        n = len(members)
        im = ax.imshow(sub, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_labels = [MODEL_META.get(m, {}).get("label", m) for m in members]
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(short_labels, fontsize=9)
        for i in range(n):
            for j in range(n):
                tc = "black" if sub[i, j] < 0.7 else "white"
                ax.text(j, i, f"{sub[i,j]:.3f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=tc)
        ax.set_title(fam, fontsize=11, fontweight="bold")

    fig.suptitle("Within-Family CKA Heatmaps", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = plots_dir / "cka_within_family.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


# ---------------------------------------------------------------------------
# Plot 9: Audio model CKA with each text LLM — strip / dot plot
# ---------------------------------------------------------------------------

def plot_audio_vs_text_strip(results: dict, plots_dir: Path):
    """
    Each row is a text LLM. Dots show CKA with each audio model.
    Lets you see at a glance which audio model is most text-LLM-like.
    """
    cka_matrix = np.array(results.get("cka_matrix", []))
    names = list(results.get("embedding_shapes", {}).keys())
    if cka_matrix.size == 0 or not names:
        return

    audio_names = [n for n in names if MODEL_META.get(n, {}).get("modality") in AUDIO_MODALITIES]
    text_names  = [n for n in names if MODEL_META.get(n, {}).get("modality") == "text"]
    if not audio_names or not text_names:
        print("  [skip] Need both audio and text models in results.json")
        return

    _style()
    fig, ax = plt.subplots(figsize=(9, max(4, len(text_names) * 0.7 + 1.5)))
    y_positions = np.arange(len(text_names))

    for yi, text_name in enumerate(text_names):
        t_idx = names.index(text_name)
        for audio_name in audio_names:
            a_idx = names.index(audio_name)
            score = cka_matrix[a_idx, t_idx]
            color = MODEL_COLORS.get(audio_name, "#555")
            ax.scatter(score, yi, color=color, s=80, zorder=3, alpha=0.9,
                       edgecolors="white", linewidths=0.6)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(text_names, fontsize=10)
    ax.set_xlabel("CKA with text LLM", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.axvline(0.3, color="#f59e0b", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(0.305, len(text_names) - 0.5, "0.3 threshold", fontsize=8,
            color="#b45309", va="top")
    ax.set_title(
        "Cross-Modal CKA: Each Audio Model vs. Each Text LLM\n"
        "(one dot per audio model per text LLM row)",
        fontsize=11, fontweight="bold",
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MODEL_COLORS.get(n, "#555"),
               markersize=9, label=n)
        for n in audio_names
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9,
              title="Audio model",
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout()
    plt.subplots_adjust(right=0.72)
    p = plots_dir / "cka_audio_vs_text_strip.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


# ---------------------------------------------------------------------------
# Plot 10: Encoder vs decoder CKA per Whisper size
# ---------------------------------------------------------------------------

def plot_encoder_vs_decoder_cka(results: dict, plots_dir: Path):
    """
    Bar chart: for each Whisper size, show CKA(encoder, decoder).
    Also shows each encoder's CKA with every text LLM for comparison.
    """
    cka_matrix = np.array(results.get("cka_matrix", []))
    names = list(results.get("embedding_shapes", {}).keys())
    if cka_matrix.size == 0 or not names:
        return

    sizes = ["base", "small", "medium", "large"]
    enc_dec_pairs = [
        (f"whisper-{s}-enc", f"whisper-{s}-dec") for s in sizes
    ]
    enc_dec_pairs = [(e, d) for e, d in enc_dec_pairs
                     if e in names and d in names]
    if not enc_dec_pairs:
        print("  [skip] No matched whisper enc/dec pairs in results")
        return

    _style()
    size_labels = [e.replace("whisper-", "").replace("-enc", "") for e, _ in enc_dec_pairs]
    scores = [cka_matrix[names.index(e), names.index(d)] for e, d in enc_dec_pairs]

    # Also get enc vs all text LLMs
    text_names = [n for n in names if MODEL_META.get(n, {}).get("modality") == "text"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: enc vs dec
    ax = axes[0]
    bar_colors = [MODEL_COLORS.get(e, "#555") for e, _ in enc_dec_pairs]
    bars = ax.bar(size_labels, scores, color=bar_colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CKA(encoder, decoder)", fontsize=10)
    ax.set_title("Within-Model: Encoder vs. Decoder CKA\n(same Whisper checkpoint)",
                 fontsize=10, fontweight="bold")

    # Right: enc CKA with each text LLM (grouped bars)
    ax2 = axes[1]
    if text_names:
        x = np.arange(len(enc_dec_pairs))
        n_text = len(text_names)
        width = 0.7 / n_text
        for ti, tname in enumerate(text_names):
            t_idx = names.index(tname)
            enc_scores = [cka_matrix[names.index(e), t_idx] for e, _ in enc_dec_pairs]
            offsets = x + (ti - (n_text - 1) / 2) * width
            ax2.bar(offsets, enc_scores, width=width * 0.9,
                    color=MODEL_COLORS.get(tname, "#555"), label=tname,
                    edgecolor="white", linewidth=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(size_labels)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("CKA(whisper-enc, text LLM)", fontsize=10)
        ax2.set_title("Whisper Encoder CKA with Each Text LLM\nby Whisper size",
                      fontsize=10, fontweight="bold")
        ax2.legend(fontsize=7, framealpha=0.9, ncol=1,
                   loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.suptitle("Whisper Encoder ↔ Decoder and ↔ Text LLM Representational Similarity",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(right=0.80)
    p = plots_dir / "encoder_vs_decoder_cka.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


# ---------------------------------------------------------------------------
# Plot 11: Model size vs effective rank scatter
# ---------------------------------------------------------------------------

def plot_size_vs_effective_rank(eigenvalues: dict, plots_dir: Path):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    audio_x, audio_y, audio_c, audio_labels = [], [], [], []
    text_x, text_y, text_c, text_labels = [], [], [], []

    for name, eigs in eigenvalues.items():
        meta = MODEL_META.get(name, {})
        size = meta.get("size_m", None)
        if size is None:
            continue
        er = effective_rank(eigs)
        color = MODEL_COLORS.get(name, "#555")
        label = name
        if meta.get("modality") in AUDIO_MODALITIES:
            audio_x.append(size); audio_y.append(er)
            audio_c.append(color); audio_labels.append(label)
        else:
            text_x.append(size); text_y.append(er)
            text_c.append(color); text_labels.append(label)

    for xs, ys, cs, ls, marker, zorder in [
        (audio_x, audio_y, audio_c, audio_labels, "^", 4),
        (text_x, text_y, text_c, text_labels, "o", 3),
    ]:
        for x, y, c, lbl in zip(xs, ys, cs, ls):
            ax.scatter(x, y, color=c, marker=marker, s=120,
                       edgecolors="#333", linewidths=0.7, zorder=zorder)
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, alpha=0.85)

    # Legend for shape
    ax.scatter([], [], marker="^", color="#888", s=90, label="Audio / speech model")
    ax.scatter([], [], marker="o", color="#888", s=90, label="Text LLM")
    ax.legend(fontsize=9, framealpha=0.9,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters (M, log scale)", fontsize=11)
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title(
        "Model Size vs. Effective Rank of Representations\n"
        "Triangle = audio/speech  ·  Circle = text LLM",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    p = plots_dir / "embedding_dim_vs_effective_rank.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


# ---------------------------------------------------------------------------
# Plot 12: Enc vs dec eigenspectrum comparison (same axes, per size)
# ---------------------------------------------------------------------------

def plot_enc_dec_eigenspectra_side_by_side(eigenvalues: dict, plots_dir: Path, max_pc: int):
    """
    2×4 grid: one column per Whisper size, top row = encoder, bottom = decoder.
    Makes it easy to see how the enc/dec spectrum shape differs within a size.
    """
    sizes = ["base", "small", "medium", "large"]
    pairs = [(f"whisper-{s}-enc", f"whisper-{s}-dec") for s in sizes]
    pairs = [(e, d) for e, d in pairs if e in eigenvalues or d in eigenvalues]
    if not pairs:
        print("  [skip] No whisper enc/dec eigenvalues found")
        return

    _style()
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.5 * len(pairs), 7),
                             sharey=False)
    if len(pairs) == 1:
        axes = axes.reshape(2, 1)

    for col, (enc_name, dec_name) in enumerate(pairs):
        size_label = enc_name.replace("whisper-", "").replace("-enc", "")
        for row, (name, role) in enumerate([(enc_name, "Encoder"), (dec_name, "Decoder")]):
            ax = axes[row, col]
            if name in eigenvalues:
                eigs = eigenvalues[name][:max_pc]
                k = np.arange(1, len(eigs) + 1)
                color = MODEL_COLORS.get(name, "#555")
                er = effective_rank(eigenvalues[name])
                ax.plot(k, eigs, color=color, linewidth=2.2)
                ax.fill_between(k, eigs, alpha=0.15, color=color)
                ax.set_title(f"{size_label} — {role}\n(eff. rank={er:.1f})",
                             fontsize=9, fontweight="bold")
            else:
                ax.set_title(f"{size_label} — {role}\n(no data)", fontsize=9)
                ax.text(0.5, 0.5, "not available", ha="center", va="center",
                        transform=ax.transAxes, color="#aaa")
            ax.set_yscale("log")
            ax.set_xlim(1, max_pc)
            ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
            if row == 1:
                ax.set_xlabel(f"PC (1–{max_pc})", fontsize=8)
            if col == 0:
                ax.set_ylabel("Frac. variance", fontsize=8)

    fig.suptitle(
        "Whisper Encoder vs. Decoder Eigenvalue Spectra — by Size",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    p = plots_dir / "whisper_enc_dec_spectra_grid.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Supplemental plots for representation_analysis.py")
    p.add_argument("--root_dir", type=str, default=".",
                   help="Root dir containing Data/ and Plots/ (default: .)")
    p.add_argument("--max_pc", type=int, default=15,
                   help="Max PCs shown on eigenspectrum x-axis (default: 15)")
    p.add_argument("--no_embeddings", action="store_true",
                   help="Skip plots that require loading .pkl embedding files")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root_dir)
    data_dir = root / "Data"
    plots_dir = root / "Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Supplemental plots — representation analysis")
    print(f"  Root : {root.resolve()}")
    print(f"  Data : {data_dir.resolve()}")
    print(f"  Plots: {plots_dir.resolve()}")
    print("=" * 60)

    # Load results.json
    print("\n[1/3] Loading results.json …")
    results = load_results(data_dir)
    eigenvalues = load_eigenvalues(results)
    available = list(eigenvalues.keys())
    print(f"  Models with eigenvalues: {available}")

    # Optionally load raw embeddings (only needed for plots that recompute PCA)
    # Currently all eigenvalue plots use pre-computed spectra from results.json,
    # so embedding loading is not needed for any of the plots below.
    # Keep the hook here in case future plots need it.

    print("\n[2/3] Generating plots …")
    print("\n── Whisper encoder/decoder variance by PC ──")
    plot_whisper_enc_variance(eigenvalues, plots_dir, args.max_pc)
    plot_whisper_dec_variance(eigenvalues, plots_dir, args.max_pc)

    print("\n── Per-family variance by PC ──")
    plot_family_variance(eigenvalues, plots_dir, args.max_pc)

    print("\n── Cumulative variance — all models ──")
    plot_cumulative_variance(eigenvalues, plots_dir, args.max_pc)

    print("\n── Participation ratio & effective rank bars ──")
    plot_participation_ratio(eigenvalues, plots_dir)

    print("\n── Within-family CKA heatmaps ──")
    plot_within_family_cka(results, plots_dir)

    print("\n── Audio vs text CKA strip plot ──")
    plot_audio_vs_text_strip(results, plots_dir)

    print("\n── Encoder vs decoder CKA ──")
    plot_encoder_vs_decoder_cka(results, plots_dir)

    print("\n── Model size vs effective rank ──")
    plot_size_vs_effective_rank(eigenvalues, plots_dir)

    print("\n── Whisper enc/dec eigenspectra grid ──")
    plot_enc_dec_eigenspectra_side_by_side(eigenvalues, plots_dir, args.max_pc)

    print("\n[3/3] Done.")
    print(f"\nAll plots saved to: {plots_dir.resolve()}")
    print("\nPlots produced:")
    for f in sorted(plots_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
