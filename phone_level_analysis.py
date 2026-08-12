#!/usr/bin/env python3
"""
Phone-Level Representation Analysis

Replicates word_level_analysis.py at phone granularity using MFA TextGrids.

Audio models: mean-pool encoder frames over phone boundary timestamps (same
              _slice_frames logic as word-level, just narrower time windows).
LLMs:         assign the parent word's contextual hidden state to every
              constituent phone — /k/, /æ/, /t/ all get the embedding of "cat".

SIL and SP (and other silence markers) are excluded.

Usage:
  python phone_level_analysis.py \\
      --ljspeech_dir /path/to/LJSpeech-1.1 \\
      --textgrid_dir /path/to/textgrids \\
      --kaldi_bin_dir /path/to/kaldi/src/nnet3bin \\
      --kaldi_model_dir /path/to/kaldi/model
"""

import argparse
import csv
import gc
import json
import logging
import pickle
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Import shared machinery from word_level_analysis
# All extractor functions work unchanged with phone records because they only
# access rec["start"], rec["end"], rec["utt_id"], rec["sentence"],
# rec["char_start"], rec["char_end"] — all present in phone records.
# ---------------------------------------------------------------------------
from word_level_analysis import (
    MODELS, MODEL_COLORS, AUDIO_MODALITIES,
    WHISPER_ENC_FPS, PARAKEET_FPS, MIMI_FPS, WAV2VEC2_FPS, KALDI_FPS,
    WHISPER_SR, MIMI_SR,
    MINIBATCH_SIZE, MINIBATCH_SEED,
    PCA_COMPONENTS, PCA_PLOT_MAX_PC,
    ensure_ljspeech, ensure_textgrids, load_ljspeech,
    extract_whisper_enc_word_embeddings  as _extract_whisper_enc,
    extract_whisper_dec_word_embeddings  as _extract_whisper_dec,
    extract_parakeet_word_embeddings     as _extract_parakeet,
    extract_mimi_word_embeddings         as _extract_mimi,
    extract_wav2vec2_word_embeddings     as _extract_wav2vec2,
    extract_kaldi_word_embeddings        as _extract_kaldi,
    extract_lm_word_embeddings           as _extract_lm,
    minibatch_cka, pca_eigenvalues, effective_rank,
    plot_similarity_histograms,
    _find_word_in_sentence,
    get_device, release_vram,
    parse_textgrid as _parse_textgrid_words,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_PHONE_DUR = 0.02   # 20 ms — minimum one Whisper frame at 50 Hz
MAX_PHONE_DUR = 1.0    # 1 s   — skip implausibly long phone intervals
SILENCE_PHONES = frozenset({
    "SIL", "sil", "SP", "sp", "spn", "SPN", "", "<eps>", "{lg}", "{ns}",
})

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("phone_level")


def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"phone_run_{ts}.log"
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return log_path


@contextmanager
def _timer(label: str):
    logger.info(f"[START] {label}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        fmt = (f"{int(h)}h {int(m)}m {s:.1f}s" if h > 0
               else f"{int(m)}m {s:.1f}s" if m > 0
               else f"{s:.2f}s")
        logger.info(f"[DONE]  {label}  ({fmt})")


# ---------------------------------------------------------------------------
# TextGrid phones-tier parser
# ---------------------------------------------------------------------------

def parse_textgrid_phones(path: Path) -> list[dict]:
    """Parse a Praat TextGrid and return non-silence phone intervals.

    Returns list of dicts: {"phone": str, "start": float, "end": float}
    Handles both short (compact) and long (verbose) TextGrid formats.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    phones = []

    for tier_name in ('"phones"', '"phone"', '"Phones"', '"Phone"'):
        m = re.search(
            rf'name = {re.escape(tier_name)}.*?(?=\bitem\s*\[|\Z)',
            text, re.DOTALL,
        )
        if m:
            tier_text = m.group(0)
            break
    else:
        logger.warning(f"No phones tier found in {path.name}")
        return phones

    interval_re = re.compile(
        r'xmin\s*=\s*([\d.eE+\-]+)\s*\n\s*xmax\s*=\s*([\d.eE+\-]+)\s*\n'
        r'\s*(?:text|intervals\s*\[\d+\].*?text)\s*=\s*"([^"]*)"',
        re.DOTALL,
    )
    for m in interval_re.finditer(tier_text):
        phone = m.group(3).strip()
        if phone in SILENCE_PHONES:
            continue
        phones.append({
            "phone": phone,
            "start": float(m.group(1)),
            "end":   float(m.group(2)),
        })

    return phones


# ---------------------------------------------------------------------------
# Build phone records
# ---------------------------------------------------------------------------

def build_phone_records(
    utterances: dict,
    textgrid_dir: Path,
    min_dur: float = MIN_PHONE_DUR,
    max_dur: float = MAX_PHONE_DUR,
    max_phones: int = None,
) -> list[dict]:
    """Combine LJSpeech utterances with TextGrid phone boundaries.

    Returns a list of phone records:
      {utt_id, phone, start, end, word, sentence, char_start, char_end}

    char_start / char_end are the *parent word's* character positions in
    the sentence, so that text models assign the word-level hidden state
    to every constituent phone (e.g. /k/, /ae/, /t/ all get "cat").
    """
    records = []
    missing_tg = 0
    skipped_no_parent = 0

    for utt_id, utt in tqdm(utterances.items(), desc="Building phone records", unit="utt"):
        tg_path = textgrid_dir / f"{utt_id}.TextGrid"
        if not tg_path.exists():
            missing_tg += 1
            continue

        phone_intervals = parse_textgrid_phones(tg_path)
        word_intervals  = _parse_textgrid_words(tg_path)
        if not phone_intervals or not word_intervals:
            continue

        sentence = utt["text"].lower()

        # Build word spans with char offsets (tracking repeated words)
        word_occ: dict[str, int] = {}
        word_spans = []
        for wi in word_intervals:
            w = wi["word"].lower()
            occ = word_occ.get(w, 0)
            word_occ[w] = occ + 1
            cs, ce = _find_word_in_sentence(w, sentence, occurrence=occ)
            word_spans.append({
                "word": w, "start": wi["start"], "end": wi["end"],
                "char_start": cs, "char_end": ce,
            })

        for pi in phone_intervals:
            dur = pi["end"] - pi["start"]
            if dur < min_dur or dur > max_dur:
                continue

            # Link phone to parent word via midpoint containment
            mid = (pi["start"] + pi["end"]) / 2
            parent = None
            for ws in word_spans:
                if ws["start"] <= mid < ws["end"]:
                    parent = ws
                    break
            if parent is None:
                skipped_no_parent += 1
                continue

            records.append({
                "utt_id":     utt_id,
                "phone":      pi["phone"],
                "start":      pi["start"],
                "end":        pi["end"],
                "word":       parent["word"],
                "sentence":   sentence,
                "char_start": parent["char_start"],
                "char_end":   parent["char_end"],
            })

            if max_phones and len(records) >= max_phones:
                logger.info(f"Reached max_phones={max_phones}, stopping early")
                return records

    if missing_tg:
        logger.warning(f"{missing_tg} utterances had no TextGrid file")
    if skipped_no_parent:
        logger.warning(f"{skipped_no_parent} phones had no parent word (skipped)")
    logger.info(f"Built {len(records):,} phone records from {len(utterances):,} utterances")
    return records


# ---------------------------------------------------------------------------
# Plotting (phone-level versions with correct filenames / titles)
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_cka_heatmap_phone(cka_matrix: np.ndarray, names: list, plots_dir: Path,
                            cka_results: dict = None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.0), max(6.5, len(names) * 0.9)))
    cmap = LinearSegmentedColormap.from_list("cka", ["#FFFFFF", "#A5D6A7", "#2E7D32"], N=256)
    im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA  (0 = no similarity,  1 = identical)", fontsize=9)

    n = len(names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = cka_matrix[i, j]
            tc = "black" if val < 0.7 else "white"
            if cka_results is not None and i != j:
                key = f"{names[min(i,j)]} vs {names[max(i,j)]}"
                std = cka_results.get(key, {}).get("ci95", None)
                label = f"{val:.3f}\n±{std:.3f}" if std is not None else f"{val:.3f}"
            else:
                label = f"{val:.3f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color=tc)

    audio_idx = [i for i, nm in enumerate(names)
                 if MODELS.get(nm, {}).get("modality") in AUDIO_MODALITIES]
    if audio_idx:
        boundary = max(audio_idx) + 0.5
        ax.axhline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.axvline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_title("Phone-Level Pairwise Minibatch Linear CKA\n"
                 "(LJSpeech + MFA TextGrid phone alignments)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    path = plots_dir / "phone_cka_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_eigenspectra_phone(eigenvalues: dict, plots_dir: Path,
                             max_pc: int = PCA_PLOT_MAX_PC):
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, eigs_full in eigenvalues.items():
        eigs = eigs_full[:max_pc]
        ax.plot(np.arange(1, len(eigs) + 1), eigs,
                color=MODEL_COLORS.get(name, "#555555"),
                linewidth=2, label=name, alpha=0.85)
    ax.set_xlabel(f"Principal Component (1–{max_pc})", fontsize=11)
    ax.set_ylabel("Fraction of Variance", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlim(1, max_pc)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"Phone-Level PCA Eigenvalue Spectra  ·  first {max_pc} PCs",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = plots_dir / "phone_eigenspectra_overlay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_effective_rank_phone(eigenvalues: dict, plots_dir: Path):
    _apply_style()
    names = list(eigenvalues.keys())
    ranks  = [effective_rank(eigenvalues[n]) for n in names]
    colors = [MODEL_COLORS.get(n, "#555555") for n in names]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    bars = ax.bar(range(len(names)), ranks, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("Phone-Level Effective Rank per Model", fontsize=12, fontweight="bold")
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{rank:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = plots_dir / "phone_effective_rank.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------

def save_summary_tables_phone(cka_matrix: np.ndarray, cka_results: dict,
                               names: list, eigenvalues: dict,
                               embeddings: dict, data_dir: Path):
    path1 = data_dir / "phone_cka_matrix.csv"
    with open(path1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, row_name in enumerate(names):
            w.writerow([row_name] + [f"{cka_matrix[i,j]:.6f}" for j in range(len(names))])
    logger.info(f"  Saved → {path1}")

    path2 = data_dir / "phone_cka_minibatch_std.csv"
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "cka_score", "ci95"])
        for pair, vals in cka_results.items():
            w.writerow([pair, f"{vals['score']:.6f}", f"{vals['ci95']:.6f}"])
    logger.info(f"  Saved → {path2}")

    path3 = data_dir / "phone_pca_summary.csv"
    with open(path3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "arch", "embedding_dim",
                    "effective_rank", "var_pc1", "var_top5", "var_top10"])
        for name, eigs in eigenvalues.items():
            dim = embeddings[name].shape[1] if name in embeddings else "—"
            w.writerow([
                name, MODELS[name]["params"], MODELS[name]["arch"], dim,
                f"{effective_rank(eigs):.2f}",
                f"{eigs[0]:.4f}", f"{eigs[:5].sum():.4f}", f"{eigs[:10].sum():.4f}",
            ])
    logger.info(f"  Saved → {path3}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phone-level cross-modal CKA on LJSpeech + MFA TextGrids"
    )
    p.add_argument("--textgrid_dir", default=None, type=Path,
                   help="Directory containing .TextGrid files. Auto-downloaded from Kaggle if absent.")
    p.add_argument("--ljspeech_dir", default=None, type=Path,
                   help="LJSpeech-1.1 root (must contain metadata.csv and wavs/). "
                        "Auto-downloaded if absent.")
    p.add_argument("--root_dir", default=".", type=Path,
                   help="Root directory for large files: PhoneData/ pkl embeddings, "
                        "phone_records.json, LJSpeech, textgrids. "
                        "Can be external storage (e.g. /dpluth-data) to save disk space.")
    p.add_argument("--output_dir", default=".", type=Path,
                   help="Directory for small outputs: results JSON, CSV tables, plots, logs. "
                        "Defaults to current directory (the repo root).")
    p.add_argument("--max_phones", default=None, type=int,
                   help="Cap total phone records (useful for testing)")
    p.add_argument("--batch_size", default=2048, type=int,
                   help="Minibatch size for CKA")
    p.add_argument("--whisper_batch_size",     default=64,  type=int)
    p.add_argument("--whisper_dec_batch_size", default=32,  type=int)
    p.add_argument("--parakeet_batch_size",    default=32,  type=int)
    p.add_argument("--mimi_batch_size",        default=64,  type=int)
    p.add_argument("--wav2vec2_batch_size",    default=64,  type=int)
    p.add_argument("--kaldi_bin_dir",    default=None, type=Path,
                   help="Path to Kaldi bin dir (contains nnet3-compute)")
    p.add_argument("--kaldi_model_dir",  default=None, type=Path,
                   help="Path to Kaldi model dir (contains final_hidden.mdl)")
    p.add_argument("--kaldi_ivector_dim", default=100, type=int)
    p.add_argument("--kaldi_batch_size",  default=500, type=int)
    p.add_argument("--kaldi_use_gpu", default="no", choices=["no", "yes", "wait"])
    p.add_argument("--lm_batch_size", default=32, type=int)
    p.add_argument("--skip_extraction", action="store_true",
                   help="Skip extraction even if no cache exists")
    p.add_argument("--pca_components",   default=50, type=int)
    p.add_argument("--pca_plot_max_pc",  default=10, type=int)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root = Path(args.root_dir)
    output_root = Path(args.output_dir)
    data_dir        = root        / "PhoneData"  # large files: pkl, phone_records.json
    output_data_dir = output_root / "PhoneData"  # small results: JSON, CSV
    plots_dir       = output_root / "PhonePlots"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_path = _setup_logging(output_root / "logs")
    logger.info("=" * 60)
    logger.info("Phone-Level Representation Analysis — run started")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("=" * 60)

    device = get_device()

    # ------------------------------------------------------------------
    # 0. Ensure datasets are present
    # ------------------------------------------------------------------
    # Default to the WordData/ directories so the already-downloaded LJSpeech
    # and TextGrids from word_level_analysis.py are reused automatically.
    word_data_dir = root / "WordData"
    ljspeech_dir = args.ljspeech_dir or (word_data_dir / "LJSpeech-1.1")
    try:
        ljspeech_dir = ensure_ljspeech(ljspeech_dir)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    textgrid_dir = args.textgrid_dir or (word_data_dir / "textgrids")
    try:
        textgrid_dir = ensure_textgrids(textgrid_dir)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load LJSpeech + build phone records
    # ------------------------------------------------------------------
    cache_suffix = f"_max{args.max_phones}" if args.max_phones else ""
    phone_records_path = data_dir / f"phone_records{cache_suffix}.json"

    if phone_records_path.exists():
        logger.info(f"Loading cached phone records from {phone_records_path}")
        with open(phone_records_path) as f:
            phone_records = json.load(f)
        logger.info(f"Loaded {len(phone_records):,} phone records")
        utterances = load_ljspeech(ljspeech_dir)
    else:
        with _timer("Load LJSpeech"):
            utterances = load_ljspeech(ljspeech_dir)
        with _timer("Build phone records"):
            phone_records = build_phone_records(
                utterances, textgrid_dir, max_phones=args.max_phones,
            )
        with open(phone_records_path, "w") as f:
            json.dump(phone_records, f)
        logger.info(f"Cached phone records → {phone_records_path}")

    N = len(phone_records)
    logger.info(f"Total phone tokens: {N:,}")

    # ------------------------------------------------------------------
    # 2. Extract / load embeddings
    # ------------------------------------------------------------------
    embeddings = {}
    for model_name, cfg in MODELS.items():
        cache_path = data_dir / f"phone_embeddings_{model_name}.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached phone embeddings: {model_name}")
            with open(cache_path, "rb") as f:
                embeddings[model_name] = pickle.load(f)
            logger.info(f"  Shape: {embeddings[model_name].shape}")
            continue

        if args.skip_extraction:
            logger.warning(f"--skip_extraction but no cache for {model_name}, skipping")
            continue

        logger.info("=" * 60)
        logger.info(f"Extracting phone embeddings: {model_name}  "
                    f"[{cfg['params']} | {cfg['arch']}]")
        logger.info("=" * 60)

        with _timer(f"Extract {model_name}"):
            try:
                modality = cfg["modality"]
                if modality == "audio-whisper-enc":
                    emb = _extract_whisper_enc(
                        model_name, cfg["hf_id"], phone_records, utterances,
                        device, fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.whisper_batch_size,
                        checkpoint_dir=data_dir,
                    )
                elif modality == "audio-whisper-dec":
                    emb = _extract_whisper_dec(
                        model_name, cfg["hf_id"], phone_records, utterances,
                        device, target_sr=cfg["target_sr"],
                        batch_size=args.whisper_dec_batch_size,
                        checkpoint_dir=data_dir,
                    )
                elif modality == "audio-parakeet":
                    emb = _extract_parakeet(
                        cfg["hf_id"], phone_records, utterances,
                        device, fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.parakeet_batch_size,
                        checkpoint_dir=data_dir,
                    )
                elif modality == "audio-mimi":
                    emb = _extract_mimi(
                        cfg["hf_id"], phone_records, utterances,
                        device, fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.mimi_batch_size,
                        checkpoint_dir=data_dir,
                    )
                elif modality == "audio-wav2vec2":
                    emb = _extract_wav2vec2(
                        model_name, cfg["hf_id"], phone_records, utterances,
                        device, fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.wav2vec2_batch_size,
                        checkpoint_dir=data_dir,
                    )
                elif modality == "audio-kaldi":
                    if not args.kaldi_model_dir or not args.kaldi_bin_dir:
                        logger.warning(
                            f"Skipping {model_name}: --kaldi_model_dir and "
                            "--kaldi_bin_dir are required for audio-kaldi models."
                        )
                        continue
                    emb = _extract_kaldi(
                        model_name,
                        kaldi_model_dir=args.kaldi_model_dir,
                        kaldi_bin_dir=args.kaldi_bin_dir,
                        word_records=phone_records,
                        utterances=utterances,
                        fps=cfg["fps"],
                        target_sr=cfg["target_sr"],
                        ivector_dim=args.kaldi_ivector_dim,
                        batch_size=args.kaldi_batch_size,
                        checkpoint_dir=data_dir,
                        use_gpu=args.kaldi_use_gpu,
                        model_filename=cfg.get("kaldi_model_file", "final_hidden.mdl"),
                    )
                else:  # text LMs
                    emb = _extract_lm(
                        model_name, cfg["hf_id"], phone_records,
                        device, batch_size=args.lm_batch_size,
                        checkpoint_dir=data_dir,
                    )
            except Exception as e:
                logger.error(f"Extraction failed for {model_name}: {e}. Skipping.")
                release_vram(f"after {model_name} failure")
                continue

        n_bad = (~np.isfinite(emb)).sum()
        if n_bad > 0:
            logger.warning(f"{model_name}: {n_bad} non-finite values")

        embeddings[model_name] = emb
        with open(cache_path, "wb") as f:
            pickle.dump(emb, f)
        logger.info(f"Cached → {cache_path}  ({cache_path.stat().st_size / 1024**2:.1f} MB)")

    names = list(embeddings.keys())
    logger.info(f"Models with embeddings: {names}")

    # Audio no longer needed — free ~13 GB before loading all embeddings for CKA
    del utterances

    # ------------------------------------------------------------------
    # 2b. Align: keep only phones where every model produced an embedding
    # ------------------------------------------------------------------
    if embeddings:
        valid_mask = np.ones(N, dtype=bool)
        for name, emb in embeddings.items():
            valid_mask &= np.linalg.norm(emb, axis=1) > 1e-10

        n_valid   = int(valid_mask.sum())
        n_dropped = N - n_valid
        frac_dropped = n_dropped / N
        logger.info(
            f"Phone alignment: {n_valid:,} / {N:,} phones valid across all models "
            f"({n_dropped:,} dropped, {frac_dropped:.2%})"
        )
        if frac_dropped > 0.02:
            raise RuntimeError(
                f"{frac_dropped:.2%} of phones ({n_dropped:,}/{N:,}) dropped — "
                f"exceeds 2% threshold. Per-model zero counts:\n" +
                "\n".join(
                    f"  {name}: {int((np.linalg.norm(emb, axis=1) <= 1e-10).sum()):,} zeros"
                    for name, emb in embeddings.items()
                )
            )
        if n_dropped > 0:
            for name in list(embeddings.keys()):
                embeddings[name] = embeddings[name][valid_mask]
            N = n_valid
            logger.info(f"All embedding matrices trimmed to {N:,} aligned phones")

    # Downcast to float16 before PCA/CKA — halves the ~85 GB base footprint.
    # Both pca_eigenvalues and minibatch_cka cast to float64 internally, so
    # precision is not lost in the actual computations.
    for name in list(embeddings.keys()):
        embeddings[name] = embeddings[name].astype(np.float16)
    gc.collect()
    logger.info("Embeddings downcast to float16 for PCA/CKA phase")

    # ------------------------------------------------------------------
    # 3. PCA eigenvalue analysis
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PCA eigenvalue analysis")
    logger.info("=" * 60)
    eigenvalues = {}
    with _timer("PCA"):
        for name, X in embeddings.items():
            eigs = pca_eigenvalues(X, n_components=args.pca_components)
            eigenvalues[name] = eigs
            er = effective_rank(eigs)
            logger.info(f"  {name:<22}  dim={X.shape[1]:>5}  eff_rank={er:.1f}")

    with _timer("PCA plots"):
        plot_eigenspectra_phone(eigenvalues, plots_dir, max_pc=args.pca_plot_max_pc)
        plot_effective_rank_phone(eigenvalues, plots_dir)

    with _timer("Similarity histograms"):
        plot_similarity_histograms(embeddings, plots_dir, prefix="phone_")

    # ------------------------------------------------------------------
    # 4. Pairwise minibatch CKA
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Pairwise phone-level minibatch CKA  (batch_size={args.batch_size}, N={N:,})")
    logger.info("=" * 60)

    n = len(names)
    n_pairs = n * (n - 1) // 2
    cka_matrix = np.zeros((n, n))
    cka_results = {}

    with _timer("Pairwise CKA"):
        pair_bar = tqdm(total=n_pairs, desc="CKA pairs", unit="pair")
        for i in range(n):
            for j in range(n):
                if i == j:
                    cka_matrix[i, j] = 1.0
                    continue
                if j < i:
                    cka_matrix[i, j] = cka_matrix[j, i]
                    continue
                t0 = time.perf_counter()
                score, ci95 = minibatch_cka(embeddings[names[i]], embeddings[names[j]],
                                             batch_size=args.batch_size)
                cka_matrix[i, j] = score
                key = f"{names[i]} vs {names[j]}"
                cka_results[key] = {"score": score, "ci95": ci95}
                logger.info(f"  CKA({names[i]}, {names[j]}) = {score:.4f}  "
                            f"(95% CI ±{ci95:.4f}, {time.perf_counter()-t0:.1f}s)")
                pair_bar.update(1)
        pair_bar.close()

    with _timer("CKA plots"):
        plot_cka_heatmap_phone(cka_matrix, names, plots_dir, cka_results=cka_results)

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    with _timer("Save tables"):
        save_summary_tables_phone(cka_matrix, cka_results, names, eigenvalues, embeddings, output_data_dir)

    results_json = {
        "n_phones": N,
        "models": {k: {kk: vv for kk, vv in v.items() if kk not in ("fps", "target_sr")}
                   for k, v in MODELS.items()},
        "embedding_shapes": {k: list(v.shape) for k, v in embeddings.items()},
        "cka_scores": cka_results,
        "cka_matrix": cka_matrix.tolist(),
        "eigenvalue_spectra": {k: v.tolist() for k, v in eigenvalues.items()},
        "effective_ranks": {k: effective_rank(v) for k, v in eigenvalues.items()},
    }
    json_path = output_data_dir / "phone_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results JSON → {json_path}")
    logger.info("Run complete.")


if __name__ == "__main__":
    main()
