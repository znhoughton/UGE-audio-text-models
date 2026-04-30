#!/usr/bin/env python3
"""
Mozilla Common Voice English Word-Level Representation Analysis:
Cross-Speaker Minibatch CKA & PCA

Parallel to mls_word_level_analysis.py but uses Mozilla Common Voice 17.0
(english config) and performs cross-speaker CKA analysis.

Cross-speaker CKA design
------------------------
CKA uses a word-type mask so that only same-word different-speaker pairs
drive the HSIC score.  Concretely, within each minibatch the kernel matrices
K and L are zeroed out for all (i, j) pairs where word_records[i]["word"] !=
word_records[j]["word"] before computing the unbiased HSIC estimator.  This
ensures that the CKA score reflects representational similarity specifically
for the same lexical item produced by different speakers, rather than
similarity across arbitrary word pairs.

Pipeline
--------
  Stage 0 — Sample & download
      Reads a local Mozilla Common Voice validated.tsv file, filters for
      sentences spoken by >= min_speakers distinct speakers, samples up to
      n_utterances, then copies audio (MP3 → 16 kHz mono WAV) and writes
      .lab files, transcripts.json, and metainfo.json
      (containing speaker_id and sentence fields).

  Stage 1 — CTC forced alignment
      Runs ctc-forced-aligner on the downloaded corpus to produce one
      TextGrid per utterance (words tier only).  The ONNX model is downloaded
      automatically on first run.

  Stage 2 — Build word records
      Parses TextGrids and builds word-level records with character offsets.

  Stage 3 — Embed & analyse
      Extracts word-level embeddings from each model (Whisper enc/dec,
      Parakeet, Mimi, text LMs), then runs cross-speaker minibatch CKA
      (with word-type mask) and PCA.

All outputs use the prefix  mcv_  and live under  MCVData/  and  MCVPlots/
so nothing from other runs is overwritten.

Usage
-----
  # Full pipeline (download + align + embed + analyse):
  python mcv_word_level_analysis_cross_speakers.py

  # Skip download if audio already present:
  python mcv_word_level_analysis_cross_speakers.py --mcv_dir /path/to/mcv_sample

  # Skip alignment if TextGrids already present:
  python mcv_word_level_analysis_cross_speakers.py --skip_mfa

  # Skip extraction if embeddings already cached:
  python mcv_word_level_analysis_cross_speakers.py --skip_extraction

  # Multi-run CKA for variability measurement:
  python mcv_word_level_analysis_cross_speakers.py --num_cka_runs 5

Dependencies
------------
  pip install datasets soundfile ctc-forced-aligner onnxruntime
"""

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import time

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
    MimiModel,
    ParakeetForCTC,
    WhisperModel,
    WhisperProcessor,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MCV_SR           = 16000    # MCV audio resampled to 16 kHz
WHISPER_SR       = 16000
MIMI_SR          = 24000

WHISPER_ENC_FPS  = 50.0
PARAKEET_FPS     = 12.5
MIMI_FPS         = 12.5

MIN_WORD_DUR     = 0.05
MAX_WORD_DUR     = 3.0
MIN_WORD_ALPHA   = 2

CHECKPOINT_EVERY = 200
MINIBATCH_SIZE   = 2048
MINIBATCH_SEED   = 42
MAX_TEXT_TOKENS  = 256
PCA_COMPONENTS   = 50
PCA_PLOT_MAX_PC  = 10


# ---------------------------------------------------------------------------
# Model registry  (identical to mls_word_level_analysis.py)
# ---------------------------------------------------------------------------

MODELS = {
    "whisper-base-enc": {
        "hf_id": "openai/whisper-base",
        "modality": "audio-whisper-enc",
        "params": "74M", "arch": "Whisper encoder", "corpus": "680k hrs audio",
        "fps": WHISPER_ENC_FPS, "target_sr": WHISPER_SR,
    },
    "whisper-small-enc": {
        "hf_id": "openai/whisper-small",
        "modality": "audio-whisper-enc",
        "params": "244M", "arch": "Whisper encoder", "corpus": "680k hrs audio",
        "fps": WHISPER_ENC_FPS, "target_sr": WHISPER_SR,
    },
    "whisper-medium-enc": {
        "hf_id": "openai/whisper-medium",
        "modality": "audio-whisper-enc",
        "params": "769M", "arch": "Whisper encoder", "corpus": "680k hrs audio",
        "fps": WHISPER_ENC_FPS, "target_sr": WHISPER_SR,
    },
    "whisper-large-enc": {
        "hf_id": "openai/whisper-large-v3",
        "modality": "audio-whisper-enc",
        "params": "1550M", "arch": "Whisper encoder", "corpus": "680k hrs audio",
        "fps": WHISPER_ENC_FPS, "target_sr": WHISPER_SR,
    },
    "whisper-base-dec": {
        "hf_id": "openai/whisper-base",
        "modality": "audio-whisper-dec",
        "params": "74M", "arch": "Whisper decoder", "corpus": "680k hrs audio",
        "target_sr": WHISPER_SR,
    },
    "whisper-small-dec": {
        "hf_id": "openai/whisper-small",
        "modality": "audio-whisper-dec",
        "params": "244M", "arch": "Whisper decoder", "corpus": "680k hrs audio",
        "target_sr": WHISPER_SR,
    },
    "whisper-medium-dec": {
        "hf_id": "openai/whisper-medium",
        "modality": "audio-whisper-dec",
        "params": "769M", "arch": "Whisper decoder", "corpus": "680k hrs audio",
        "target_sr": WHISPER_SR,
    },
    "whisper-large-dec": {
        "hf_id": "openai/whisper-large-v3",
        "modality": "audio-whisper-dec",
        "params": "1550M", "arch": "Whisper decoder", "corpus": "680k hrs audio",
        "target_sr": WHISPER_SR,
    },
    "parakeet-ctc-0.6b": {
        "hf_id": "nvidia/parakeet-ctc-0.6b",
        "modality": "audio-parakeet",
        "params": "600M", "arch": "FastConformer-CTC", "corpus": "Granary (64k hrs English)",
        "fps": PARAKEET_FPS, "target_sr": WHISPER_SR,
    },
    "mimi": {
        "hf_id": "kyutai/mimi",
        "modality": "audio-mimi",
        "params": "~85M", "arch": "Conv+Transformer codec", "corpus": "Moshi training set",
        "fps": MIMI_FPS, "target_sr": MIMI_SR,
    },
    "babylm-125m": {
        "hf_id": "znhoughton/opt-babylm-125m-20eps-seed964",
        "modality": "text", "params": "125M", "arch": "OPT", "corpus": "BabyLM (~100M tokens)",
    },
    "opt-125m": {
        "hf_id": "facebook/opt-125m",
        "modality": "text", "params": "125M", "arch": "OPT", "corpus": "~180B tokens",
    },
    "babylm-350m": {
        "hf_id": "znhoughton/opt-babylm-350m-20eps-seed964",
        "modality": "text", "params": "350M", "arch": "OPT", "corpus": "BabyLM (~100M tokens)",
    },
    "babylm-1.3b": {
        "hf_id": "znhoughton/opt-babylm-1.3b-20eps-seed964",
        "modality": "text", "params": "1.3B", "arch": "OPT", "corpus": "BabyLM (~100M tokens)",
    },
    "pythia-160m": {
        "hf_id": "EleutherAI/pythia-160m",
        "modality": "text", "params": "160M", "arch": "GPT-NeoX", "corpus": "The Pile (300B tokens)",
    },
    "olmo-7b": {
        "hf_id": "allenai/OLMo-2-1124-7B",
        "modality": "text", "params": "7B", "arch": "OLMo-2", "corpus": "Dolma",
    },
    "pythia-6.9b": {
        "hf_id": "EleutherAI/pythia-6.9b",
        "modality": "text", "params": "6.9B", "arch": "GPT-NeoX", "corpus": "The Pile (300B tokens)",
    },
}

AUDIO_MODALITIES = {"audio-whisper-enc", "audio-whisper-dec", "audio-parakeet", "audio-mimi"}

MODEL_COLORS = {
    "whisper-base-enc":    "#B3E5FC",
    "whisper-small-enc":   "#4FC3F7",
    "whisper-medium-enc":  "#0288D1",
    "whisper-large-enc":   "#01579B",
    "whisper-base-dec":    "#E1F5FE",
    "whisper-small-dec":   "#81D4FA",
    "whisper-medium-dec":  "#039BE5",
    "whisper-large-dec":   "#0277BD",
    "parakeet-ctc-0.6b":   "#880E4F",
    "mimi":                "#D84315",
    "babylm-125m":         "#E65100",
    "opt-125m":            "#FFCCBC",
    "babylm-350m":         "#FB8C00",
    "babylm-1.3b":         "#F9A825",
    "olmo-7b":             "#2E7D32",
    "pythia-160m":         "#CE93D8",
    "pythia-6.9b":         "#6A1B9A",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("mcv_word_level")


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"mcv_run_{timestamp}.log"
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
def timer(label: str):
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


def log_gpu_memory(label: str = ""):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    peak      = torch.cuda.max_memory_allocated() / 1024 ** 3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    tag = f"[{label}] " if label else ""
    logger.debug(f"{tag}GPU memory — allocated: {allocated:.2f} GB  "
                 f"reserved: {reserved:.2f} GB  peak: {peak:.2f} GB  total: {total:.2f} GB")


def release_vram(label: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log_gpu_memory(f"after release [{label}]")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name}  ({props.total_memory / 1024**3:.1f} GB VRAM)")
        return torch.device("cuda")
    logger.warning("No GPU detected — running on CPU (will be very slow)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage 0 — Prepare MCV sample from local files
# ---------------------------------------------------------------------------

def _make_utt_id(client_id: str, sentence: str) -> str:
    """Stable filesystem-safe utterance ID from speaker + sentence."""
    spk_hash = hashlib.md5(client_id.encode()).hexdigest()[:10]
    sen_hash = hashlib.md5(sentence.encode()).hexdigest()[:10]
    return f"spk{spk_hash}_sen{sen_hash}"


def download_mcv_sample(
    mcv_source_dir: Path,
    mcv_dir: Path,
    n_utterances: int = None,
    n_words: int = 250_000,
    seed: int = 42,
    min_speakers: int = 250,
    tsv_file: str = "validated.tsv",
    dry_run: bool = False,
) -> Path:
    """Prepare an MCV sample from a locally downloaded Common Voice release.

    Reads <mcv_source_dir>/<tsv_file> (TSV with client_id / path / sentence
    columns) and <mcv_source_dir>/clips/<filename>.mp3.

    Steps:
      1. Read TSV → find sentences spoken by >= min_speakers unique speakers.
      2. Shuffle and take utterances until n_words word tokens are reached
         (or n_utterances rows if n_utterances is set instead).
      3. Convert each MP3 to 16 kHz mono WAV and write .lab transcript.

    Saves:
      <mcv_dir>/wavs/<utt_id>.wav   — 16 kHz mono WAV
      <mcv_dir>/lab/<utt_id>.lab    — plain-text transcript for alignment
      <mcv_dir>/transcripts.json    — {utt_id: text} mapping
      <mcv_dir>/metainfo.json       — {utt_id: {speaker_id, sentence}}

    Returns mcv_dir.
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("Install with: pip install pandas")
    try:
        import soundfile as sf
    except ImportError:
        raise RuntimeError("Install with: pip install soundfile")

    transcripts_path = mcv_dir / "transcripts.json"
    wavs_dir         = mcv_dir / "wavs"

    if transcripts_path.exists():
        existing = list(wavs_dir.glob("*.wav")) if wavs_dir.exists() else []
        target_label = f"{n_utterances:,} utts" if n_utterances else f"{n_words:,} words"
        if existing:
            logger.info(f"MCV sample already present ({len(existing):,} wavs) — skipping")
            return mcv_dir
        logger.info(f"Partial download ({len(existing):,} wavs, target {target_label}) — re-preparing")

    mcv_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(exist_ok=True)
    lab_dir = mcv_dir / "lab"
    lab_dir.mkdir(exist_ok=True)

    tsv_path   = mcv_source_dir / tsv_file
    clips_dir  = mcv_source_dir / "clips"

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"clips/ directory not found: {clips_dir}")

    # ------------------------------------------------------------------ #
    # Step 1: read TSV, find multi-speaker sentences                      #
    # ------------------------------------------------------------------ #
    logger.info(f"Reading {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t",
                     usecols=["client_id", "path", "sentence"],
                     low_memory=False)
    df["sentence"] = df["sentence"].str.strip().str.lower()
    df = df.dropna(subset=["client_id", "path", "sentence"])
    logger.info(f"  {len(df):,} rows, {df['client_id'].nunique():,} speakers, "
                f"{df['sentence'].nunique():,} unique sentences")

    speakers_per_sentence = df.groupby("sentence")["client_id"].nunique()
    qualifying = speakers_per_sentence[speakers_per_sentence >= min_speakers].index
    df = df[df["sentence"].isin(qualifying)].reset_index(drop=True)

    logger.info(f"  {len(qualifying):,} sentences with >= {min_speakers} speakers "
                f"→ {len(df):,} qualifying rows")

    if df.empty:
        raise RuntimeError(f"No sentences with >= {min_speakers} speakers found in {tsv_path}")

    # ------------------------------------------------------------------ #
    # Step 2: sample rows by word-count target (or utterance count)       #
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(seed)
    df["_n_words"] = df["sentence"].str.split().str.len()

    if n_utterances is not None:
        df = df.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
        if len(df) > n_utterances:
            df = df.iloc[:n_utterances]
            logger.info(f"  Sampled {n_utterances:,} utterances")
        else:
            logger.info(f"  Using all {len(df):,} qualifying rows (< target {n_utterances:,})")
    else:
        # Greedily include sentences sorted by length desc, then speaker count
        # desc. Longest multi-word sentences come first; single-word sentences
        # ("yes", "nine", ...) are only included if the word target isn't yet met.
        sent_stats = (
            df.groupby("sentence")
            .agg(n_speakers=("client_id", "nunique"), n_words_per_sent=("_n_words", "first"))
            .sort_values(["n_words_per_sent", "n_speakers"], ascending=[False, False])
        )

        selected = []
        total_words = 0
        for sentence, row in sent_stats.iterrows():
            rows = df[df["sentence"] == sentence]
            selected.append(rows)
            total_words += int(rows["_n_words"].sum())
            if total_words >= n_words:
                break

        df = pd.concat(selected).sample(
            frac=1, random_state=int(rng.integers(0, 2**31))
        ).reset_index(drop=True)
        logger.info(f"  Selected {len(selected):,} sentences "
                    f"(longest first, then most-shared) → {len(df):,} utterances")

    df = df.drop(columns=["_n_words"])
    actual_words = int(df["sentence"].str.split().str.len().sum())
    logger.info(f"  Word tokens: {actual_words:,} "
                f"(avg {actual_words / len(df):.1f} words/utterance, "
                f"{df['client_id'].nunique():,} speakers)")

    # Cross-speaker overlap check
    spk_per_sent = df.groupby("sentence")["client_id"].nunique()
    multi = (spk_per_sent >= 2).sum()
    logger.info(f"  Cross-speaker overlap: {multi:,}/{len(spk_per_sent):,} sentences "
                f"have >=2 speakers in sample "
                f"(avg {spk_per_sent.mean():.1f}, median {spk_per_sent.median():.0f} speakers/sentence)")

    if dry_run:
        logger.info("--dry_run: skipping audio conversion")
        return mcv_dir

    # ------------------------------------------------------------------ #
    # Step 3: convert MP3 → 16 kHz WAV, write .lab files                 #
    # ------------------------------------------------------------------ #
    logger.info(f"Converting {len(df):,} clips to WAV ...")
    transcripts: dict = {}
    metainfo:    dict = {}
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting clips", unit="utt"):
        try:
            client_id = str(row["client_id"])
            sentence  = str(row["sentence"])
            clip_path = clips_dir / str(row["path"])
            utt_id    = _make_utt_id(client_id, sentence)

            if not clip_path.exists():
                errors += 1
                logger.warning(f"Clip not found: {clip_path}")
                continue

            waveform, sr = torchaudio.load(str(clip_path))

            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16 kHz if needed
            if sr != MCV_SR:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=MCV_SR)

            array = waveform.squeeze(0).numpy()
            sf.write(str(wavs_dir / f"{utt_id}.wav"), array, MCV_SR, subtype="PCM_16")
            (lab_dir / f"{utt_id}.lab").write_text(sentence, encoding="utf-8")

            transcripts[utt_id] = sentence
            metainfo[utt_id]    = {"speaker_id": client_id, "sentence": sentence}

        except Exception as e:
            errors += 1
            logger.warning(f"Failed for {row.get('path', '?')}: {e}")

    if errors:
        logger.warning(f"{errors} clips failed to convert")
    if not transcripts:
        raise RuntimeError("No utterances were saved — all clips failed.")

    with open(transcripts_path, "w") as f:
        json.dump(transcripts, f, indent=2)
    with open(mcv_dir / "metainfo.json", "w") as f:
        json.dump(metainfo, f, indent=2)

    n_unique_speakers = len({v["speaker_id"] for v in metainfo.values()})
    logger.info(f"Saved {len(transcripts):,} utterances from {n_unique_speakers:,} speakers → {mcv_dir}")
    return mcv_dir


# ---------------------------------------------------------------------------
# Stage 1 — CTC forced alignment
# ---------------------------------------------------------------------------

def run_ctc_alignment(
    mcv_dir: Path,
    textgrid_dir: Path,
    batch_size: int = 8,
) -> Path:
    """Align MCV utterances using ctc-forced-aligner (pure PyTorch/ONNX, no Kaldi).

    For each utterance, runs the MMS-FA ONNX model to get word-level timestamps,
    then writes a Praat TextGrid file (words tier only) so the rest of the pipeline
    — which already knows how to parse TextGrids — works unchanged.

    ctc-forced-aligner operates at 16 kHz; MCV wavs are already 16 kHz so no
    resampling is needed.  The ONNX model is downloaded automatically on first run
    (~75 MB) to ~/.cache/ctc_forced_aligner/.

    Parameters
    ----------
    mcv_dir       : directory containing wavs/ and transcripts.json
    textgrid_dir  : output directory for .TextGrid files (one per utterance)
    batch_size    : number of utterances to process before clearing GPU cache
    """
    # Full skip only if ALL utterances already have TextGrids
    textgrid_dir.mkdir(parents=True, exist_ok=True)
    existing_tg = list(textgrid_dir.glob("*.TextGrid")) if textgrid_dir.exists() else []
    transcripts_path_check = mcv_dir / "transcripts.json"
    if transcripts_path_check.exists():
        with open(transcripts_path_check) as _f:
            _n_utts = len(json.load(_f))
        if len(existing_tg) >= _n_utts * 0.95:
            logger.info(f"TextGrids already complete ({len(existing_tg):,}/{_n_utts:,}) — skipping alignment")
            return textgrid_dir

    try:
        import ctc_forced_aligner as cfa
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "ctc-forced-aligner is required.\n"
            "Install with:  pip install ctc-forced-aligner"
        )

    wavs_dir = mcv_dir / "wavs"

    # Load transcripts
    transcripts_path = mcv_dir / "transcripts.json"
    with open(transcripts_path) as f:
        transcripts = json.load(f)

    utt_ids = list(transcripts.keys())
    logger.info(f"Aligning {len(utt_ids):,} utterances with ctc-forced-aligner...")

    # Download / locate the ONNX model
    model_cache = Path.home() / ".cache" / "ctc_forced_aligner"
    model_path  = model_cache / "model.onnx"
    model_cache.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensuring ONNX alignment model at {model_path} ...")
    cfa.ensure_onnx_model(str(model_path), cfa.MODEL_URL)

    # Create ONNX session — use GPU execution provider if available
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if torch.cuda.is_available() else ["CPUExecutionProvider"])
    session = ort.InferenceSession(str(model_path), providers=providers)
    logger.info(f"ONNX session providers: {session.get_providers()}")

    # Tokenizer (same one used internally by ctc-forced-aligner)
    tokenizer = cfa.Tokenizer()

    # Build set of already-completed utterances so we can resume
    already_done = {p.stem for p in textgrid_dir.glob("*.TextGrid")}
    if already_done:
        logger.info(f"Resuming — {len(already_done):,} TextGrids already exist, skipping those")

    errors   = 0
    n_done   = len(already_done)
    pbar     = tqdm(utt_ids, desc="CTC alignment", unit="utt")

    for utt_id in pbar:
        wav_path = wavs_dir / f"{utt_id}.wav"
        tg_path  = textgrid_dir / f"{utt_id}.TextGrid"

        if utt_id in already_done or tg_path.exists():
            n_done += 1
            continue
        if not wav_path.exists():
            errors += 1
            logger.warning(f"WAV missing for {utt_id}")
            continue

        transcript = transcripts[utt_id].strip().lower()
        if not transcript:
            errors += 1
            continue

        # Normalise transcript to only contain characters in the MMS-FA vocab:
        #   a-z, apostrophe, space.  Everything else is removed or substituted.
        # Hyphens between words become spaces (e.g. "well-known" → "well known").
        transcript = transcript.replace("-", " ")
        transcript = re.sub(r"[^a-z' ]", "", transcript)
        transcript = re.sub(r" +", " ", transcript).strip()
        if not transcript:
            errors += 1
            continue

        try:
            # Load audio as float32 numpy array at 16 kHz (already the right SR)
            audio = cfa.load_audio(str(wav_path), ret_type="np")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Generate CTC emissions
            emissions, stride = cfa.generate_emissions(session, audio)

            # Tokenise transcript into char-level tokens with <star> markers
            tokens_starred, text_starred = cfa.preprocess_text(
                transcript,
                romanize=False,
                language="eng",
                split_size="word",
                star_frequency="segment",
            )

            # Align
            segments, scores, blank = cfa.get_alignments(emissions, tokens_starred, tokenizer)
            spans = cfa.get_spans(tokens_starred, segments, blank)

            # Get word-level timestamps as list of {"start", "end", "text"}
            word_stamps = cfa.postprocess_results(text_starred, spans, stride, scores)

            # Write TextGrid (words tier only — that's all parse_textgrid needs)
            _write_textgrid(tg_path, word_stamps, duration=float(len(audio)) / cfa.SAMPLING_FREQ)
            n_done += 1

        except Exception as e:
            errors += 1
            logger.warning(f"Alignment failed for {utt_id}: {e}")

        if (n_done + errors) % (batch_size * 50) == 0:
            gc.collect()

    pbar.close()
    tg_count = len(list(textgrid_dir.glob("*.TextGrid")))
    logger.info(f"CTC alignment complete: {tg_count:,} TextGrids written, {errors} errors")
    if errors > 0:
        logger.warning(f"{errors}/{len(utt_ids)} utterances failed alignment")
    return textgrid_dir


def _write_textgrid(path: Path, word_stamps: list, duration: float):
    """Write a minimal Praat TextGrid with a single words IntervalTier.

    word_stamps: list of {"start": float, "end": float, "text": str}
    """
    # Sort by start time and merge any overlapping/gap intervals with silence
    word_stamps = sorted(word_stamps, key=lambda x: x["start"])

    # Build full interval list including silences between words
    intervals = []
    cursor = 0.0
    for ws in word_stamps:
        start = round(ws["start"], 6)
        end   = round(ws["end"],   6)
        word  = ws["text"].strip()
        if not word:
            continue
        if start > cursor + 1e-6:
            intervals.append((cursor, start, ""))   # silence gap
        intervals.append((start, end, word))
        cursor = end
    if cursor < duration - 1e-6:
        intervals.append((cursor, duration, ""))    # trailing silence

    n = len(intervals)
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = 0",
        f"xmax = {duration:.6f}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "words"',
        f"        xmin = 0",
        f"        xmax = {duration:.6f}",
        f"        intervals: size = {n}",
    ]
    for i, (xmin, xmax, label) in enumerate(intervals, 1):
        lines += [
            f"        intervals [{i}]:",
            f"            xmin = {xmin:.6f}",
            f"            xmax = {xmax:.6f}",
            f'            text = "{label}"',
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Load MCV utterances
# ---------------------------------------------------------------------------

def load_mcv_utterances(mcv_dir: Path) -> dict:
    """Load pre-downloaded MCV sample.

    Returns dict: utt_id -> {audio: np.ndarray, sr: int, text: str, speaker_id: str}
    Loads metainfo.json to retrieve speaker_id for each utterance.
    """
    transcripts_path = mcv_dir / "transcripts.json"
    metainfo_path    = mcv_dir / "metainfo.json"
    wavs_dir         = mcv_dir / "wavs"

    if not transcripts_path.exists():
        raise FileNotFoundError(
            f"transcripts.json not found in {mcv_dir}. "
            "Run without --skip_download first."
        )

    with open(transcripts_path) as f:
        transcripts = json.load(f)

    metainfo = {}
    if metainfo_path.exists():
        with open(metainfo_path) as f:
            metainfo = json.load(f)

    logger.info(f"Loading {len(transcripts):,} MCV utterances from {mcv_dir}...")
    utterances = {}

    for utt_id, text in tqdm(transcripts.items(), desc="Loading MCV audio", unit="utt"):
        wav_path = wavs_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            logger.warning(f"WAV missing for {utt_id}")
            continue
        waveform, sr = torchaudio.load(str(wav_path))
        audio = waveform.squeeze(0).numpy().astype(np.float32)
        speaker_id = metainfo.get(utt_id, {}).get("speaker_id", "unknown")
        utterances[utt_id] = {"audio": audio, "sr": int(sr), "text": text,
                              "speaker_id": speaker_id}

    logger.info(f"Loaded {len(utterances):,} MCV utterances")
    return utterances


# ---------------------------------------------------------------------------
# TextGrid parser  (identical to mls_word_level_analysis.py)
# ---------------------------------------------------------------------------

def parse_textgrid(path: Path) -> list[dict]:
    """Parse a Praat TextGrid and return word-level intervals.

    Searches for tiers named "words", "word" (case-insensitive), or any tier
    whose name ends with "- words" / "- word" (MFA speaker-prefixed format).
    Returns list of {"word", "start", "end"} dicts; silences are skipped.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    words = []

    # Candidate tier name patterns in order of preference
    tier_patterns = [
        r'"words"', r'"word"', r'"Words"', r'"Word"',
        # MFA speaker-prefixed: e.g. name = "1234 - words"
        r'"[^"]*\s-\s[Ww]ords?"',
    ]

    tier_text = None
    for pat in tier_patterns:
        m = re.search(
            rf'name = {pat}.*?(?=\bitem\s*\[|\Z)',
            text, re.DOTALL,
        )
        if m:
            tier_text = m.group(0)
            break

    if tier_text is None:
        logger.warning(f"No words tier found in {path.name}")
        return words

    interval_pattern = re.compile(
        r'xmin\s*=\s*([\d.eE+\-]+)\s*\n\s*xmax\s*=\s*([\d.eE+\-]+)\s*\n'
        r'\s*(?:text|intervals\s*\[\d+\].*?text)\s*=\s*"([^"]*)"',
        re.DOTALL,
    )
    for m in interval_pattern.finditer(tier_text):
        word  = m.group(3).strip()
        start = float(m.group(1))
        end   = float(m.group(2))
        if not word or word.lower() in ("sp", "sil", "<eps>", "spn", "{lg}", "{ns}", "<unk>"):
            continue
        words.append({"word": word, "start": start, "end": end})

    return words


# ---------------------------------------------------------------------------
# Build word records
# ---------------------------------------------------------------------------

def build_word_records(
    utterances: dict,
    textgrid_dir: Path,
    min_dur: float = MIN_WORD_DUR,
    max_dur: float = MAX_WORD_DUR,
    max_words: int = None,
) -> list[dict]:
    """Build word-level records from utterances + TextGrids.

    Each record includes speaker_id from the utterance dict.
    TextGrids may live in speaker subdirectories under textgrid_dir.
    We build a flat index: utt_id -> TextGrid path.
    """
    # Index all TextGrids by stem (utterance id)
    tg_index: dict[str, Path] = {}
    for tg_path in textgrid_dir.rglob("*.TextGrid"):
        tg_index[tg_path.stem] = tg_path

    logger.info(f"Found {len(tg_index):,} TextGrids under {textgrid_dir}")

    records = []
    missing_tg = 0

    for utt_id, utt in tqdm(utterances.items(), desc="Building word records", unit="utt"):
        tg_path = tg_index.get(utt_id)
        if tg_path is None:
            missing_tg += 1
            continue

        word_intervals = parse_textgrid(tg_path)
        if not word_intervals:
            continue

        sentence = utt["text"].lower()
        word_occurrence_count: dict[str, int] = {}

        for wi in word_intervals:
            word = wi["word"].lower()
            dur  = wi["end"] - wi["start"]

            if dur < min_dur or dur > max_dur:
                continue
            if sum(c.isalpha() for c in word) < MIN_WORD_ALPHA:
                continue

            occ = word_occurrence_count.get(word, 0)
            word_occurrence_count[word] = occ + 1
            char_start, char_end = _find_word_in_sentence(word, sentence, occurrence=occ)

            records.append({
                "utt_id":     utt_id,
                "word":       word,
                "start":      wi["start"],
                "end":        wi["end"],
                "sentence":   sentence,
                "char_start": char_start,
                "char_end":   char_end,
                "speaker_id": utt.get("speaker_id", "unknown"),
            })

            if max_words and len(records) >= max_words:
                logger.info(f"Reached max_words={max_words}, stopping early")
                return records

    if missing_tg > 0:
        logger.warning(f"{missing_tg} utterances had no TextGrid file")
    logger.info(f"Built {len(records):,} word records from {len(utterances):,} utterances")
    return records


# ---------------------------------------------------------------------------
# Build word type IDs for CKA masking
# ---------------------------------------------------------------------------

def build_word_type_ids(word_records: list) -> np.ndarray:
    """Map each word record to an integer word-type ID for CKA masking.

    Returns an int32 array of length len(word_records) where each element
    is the integer ID of the word type (i.e. the canonical lowercase form).
    Same word type = same integer, enabling efficient mask construction in
    minibatch_cka via broadcasting: mask = wids[:, None] == wids[None, :].
    """
    all_words = sorted(set(rec["word"] for rec in word_records))
    word_to_id = {w: i for i, w in enumerate(all_words)}
    return np.array([word_to_id[rec["word"]] for rec in word_records], dtype=np.int32)


# ---------------------------------------------------------------------------
# Word-in-sentence finder  (identical to mls_word_level_analysis.py)
# ---------------------------------------------------------------------------

def _find_word_in_sentence(word: str, sentence: str, occurrence: int = 0) -> tuple[int, int]:
    def _nth_match(pattern: str, text: str, n: int) -> tuple[int, int]:
        matches = list(re.finditer(pattern, text))
        if n < len(matches):
            m = matches[n]
            return m.start(), m.end()
        if matches:
            m = matches[-1]
            return m.start(), m.end()
        return -1, -1

    start, end = _nth_match(r'\b' + re.escape(word) + r'\b', sentence, occurrence)
    if start >= 0:
        return start, end

    clean = re.sub(r"[^a-z']", "", word)
    if clean and clean != word:
        start, end = _nth_match(r'\b' + re.escape(clean) + r'\b', sentence, occurrence)
        if start >= 0:
            return start, end

    return -1, -1


# ---------------------------------------------------------------------------
# Checkpoint helpers  (identical to mls_word_level_analysis.py, prefix mcv_)
# ---------------------------------------------------------------------------

def _load_checkpoint(path: Path, label: str) -> tuple[int, list]:
    if path and path.exists():
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        logger.info(f"Resuming {label} from batch index {ckpt['next_utt']}")
        return ckpt["next_utt"], ckpt["embeddings"]
    return 0, []


def _save_checkpoint(path: Path, embeddings: list, next_utt: int):
    if path:
        with open(path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "next_utt": next_utt}, f)


def _remove_checkpoint(path: Path):
    if path and path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Audio utilities  (identical to mls_word_level_analysis.py)
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    t = torch.from_numpy(audio).unsqueeze(0)
    t = torchaudio.functional.resample(t, orig_freq=src_sr, new_freq=dst_sr)
    return t.squeeze(0).numpy()


def _resample_utterances(utterances: dict, utt_ids: list, target_sr: int) -> str:
    key = f"audio_{target_sr}"
    to_resample = [uid for uid in utt_ids if key not in utterances[uid]]
    if to_resample:
        for uid in tqdm(to_resample, desc=f"Resampling to {target_sr} Hz", unit="utt"):
            utt = utterances[uid]
            utterances[uid][key] = _resample(utt["audio"], utt["sr"], target_sr)
    return key


def _time_to_frame(t: float, fps: float, max_frame: int) -> int:
    return min(int(t * fps), max_frame)


def _slice_frames(hidden: np.ndarray, start: float, end: float, fps: float) -> np.ndarray | None:
    T  = hidden.shape[0]
    f0 = _time_to_frame(start, fps, T)
    f1 = _time_to_frame(end,   fps, T)
    f1 = max(f0 + 1, f1)
    f1 = min(f1, T)
    if f0 >= T:
        return None
    return hidden[f0:f1].mean(axis=0)


# ---------------------------------------------------------------------------
# Shared batch helper  (identical to mls_word_level_analysis.py)
# ---------------------------------------------------------------------------

def _run_batch_audio(
    model_name, utt_ids, start_utt, batch_size,
    word_records, utterances, utt_to_words, completed_words,
    word_embeddings_list, encode_fn, fps, checkpoint_path,
    audio_key="audio_resampled",
) -> int:
    N_utt  = len(utt_ids)
    errors = 0
    pbar   = tqdm(range(start_utt, N_utt, batch_size), desc=model_name,
                  unit="batch", total=(N_utt - start_utt + batch_size - 1) // batch_size)

    for batch_start in pbar:
        batch_ids = utt_ids[batch_start : batch_start + batch_size]
        try:
            audio_arrays = [utterances[uid][audio_key] for uid in batch_ids]
            hidden_batch = encode_fn(audio_arrays)

            for b, utt_id in enumerate(batch_ids):
                hidden = hidden_batch[b]
                for word_idx in utt_to_words[utt_id]:
                    if word_idx in completed_words:
                        continue
                    rec = word_records[word_idx]
                    emb = _slice_frames(hidden, rec["start"], rec["end"], fps)
                    if emb is None:
                        errors += 1
                        continue
                    word_embeddings_list.append((word_idx, emb))
                    completed_words.add(word_idx)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _save_checkpoint(checkpoint_path, word_embeddings_list, batch_start)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch at utt {batch_start}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch at utt {batch_start}: {e}")

        next_utt = batch_start + batch_size
        if next_utt % (CHECKPOINT_EVERY * batch_size) < batch_size:
            _save_checkpoint(checkpoint_path, word_embeddings_list, next_utt)

    return errors


# ---------------------------------------------------------------------------
# Whisper encoder
# ---------------------------------------------------------------------------

def extract_whisper_enc_word_embeddings(
    model_name, model_id, word_records, utterances, device,
    fps=WHISPER_ENC_FPS, target_sr=WHISPER_SR, batch_size=64, checkpoint_dir=None,
) -> np.ndarray:
    logger.info(f"Loading Whisper encoder: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model     = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()

    utt_to_words = _group_by_utt(word_records)
    utt_ids      = list(utt_to_words.keys())
    audio_key    = _resample_utterances(utterances, utt_ids, target_sr)

    checkpoint_path = checkpoint_dir / f"mcv_{model_name}_checkpoint.pkl" if checkpoint_dir else None
    start_utt, word_embeddings_list = _load_checkpoint(checkpoint_path, model_name)
    completed_words = {idx for idx, _ in word_embeddings_list}

    def encode(audio_arrays):
        inputs   = processor(audio_arrays, sampling_rate=target_sr, return_tensors="pt")
        features = inputs["input_features"].to(device, dtype=torch.float16)
        with torch.no_grad():
            enc_out = model.encoder(features, output_hidden_states=False)
        return enc_out.last_hidden_state.float().cpu().numpy()

    errors = _run_batch_audio(
        model_name, utt_ids, start_utt, batch_size,
        word_records, utterances, utt_to_words, completed_words,
        word_embeddings_list, encode, fps, checkpoint_path, audio_key,
    )
    del model
    release_vram(model_name)
    _remove_checkpoint(checkpoint_path)
    if errors:
        logger.warning(f"{model_name}: {errors} errors")
    return _sort_embeddings(word_embeddings_list, len(word_records))


# ---------------------------------------------------------------------------
# Whisper decoder
# ---------------------------------------------------------------------------

def extract_whisper_dec_word_embeddings(
    model_name, model_id, word_records, utterances, device,
    target_sr=WHISPER_SR, batch_size=32, checkpoint_dir=None,
) -> np.ndarray:
    logger.info(f"Loading Whisper decoder: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model     = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    utt_to_words = _group_by_utt(word_records)
    utt_ids      = list(utt_to_words.keys())
    N_utt        = len(utt_ids)
    audio_key    = _resample_utterances(utterances, utt_ids, target_sr)

    checkpoint_path = checkpoint_dir / f"mcv_{model_name}_checkpoint.pkl" if checkpoint_dir else None
    start_utt, word_embeddings_list = _load_checkpoint(checkpoint_path, model_name)
    completed_words = {idx for idx, _ in word_embeddings_list}

    errors = 0
    pbar   = tqdm(range(start_utt, N_utt, batch_size), desc=model_name, unit="batch",
                  total=(N_utt - start_utt + batch_size - 1) // batch_size)

    for batch_start in pbar:
        batch_ids = utt_ids[batch_start : batch_start + batch_size]
        sentences = [word_records[utt_to_words[uid][0]]["sentence"] for uid in batch_ids]
        try:
            audio_arrays  = [utterances[uid][audio_key] for uid in batch_ids]
            audio_inputs  = processor(audio_arrays, sampling_rate=target_sr, return_tensors="pt")
            features      = audio_inputs["input_features"].to(device, dtype=torch.float16)
            with torch.no_grad():
                encoder_hidden = model.encoder(features).last_hidden_state

            text_enc = tokenizer(sentences, return_tensors="pt", truncation=True,
                                 max_length=MAX_TEXT_TOKENS, padding=True,
                                 return_offsets_mapping=True)
            offset_mapping       = text_enc.pop("offset_mapping").tolist()
            decoder_input_ids    = text_enc["input_ids"].to(device)
            decoder_attn_mask    = text_enc["attention_mask"].to(device)

            with torch.no_grad():
                dec_out = model.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attn_mask,
                    encoder_hidden_states=encoder_hidden,
                )
            hidden_batch = dec_out.last_hidden_state.float().cpu().numpy()

            for b, utt_id in enumerate(batch_ids):
                hidden = hidden_batch[b]
                om     = offset_mapping[b]
                for word_idx in utt_to_words[utt_id]:
                    if word_idx in completed_words:
                        continue
                    rec    = word_records[word_idx]
                    char_s, char_e = rec["char_start"], rec["char_end"]
                    if char_s < 0:
                        errors += 1
                        continue
                    token_indices = [t for t, (ts, te) in enumerate(om)
                                     if ts < char_e and te > char_s and ts < te]
                    if not token_indices:
                        errors += 1
                        continue
                    word_embeddings_list.append((word_idx, hidden[token_indices].mean(axis=0)))
                    completed_words.add(word_idx)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _save_checkpoint(checkpoint_path, word_embeddings_list, batch_start)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch at utt {batch_start}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch at utt {batch_start}: {e}")

        next_utt = batch_start + batch_size
        if next_utt % (CHECKPOINT_EVERY * batch_size) < batch_size:
            _save_checkpoint(checkpoint_path, word_embeddings_list, next_utt)

    del model
    release_vram(model_name)
    _remove_checkpoint(checkpoint_path)
    if errors:
        logger.warning(f"{model_name}: {errors} errors")
    return _sort_embeddings(word_embeddings_list, len(word_records))


# ---------------------------------------------------------------------------
# Parakeet
# ---------------------------------------------------------------------------

def extract_parakeet_word_embeddings(
    model_id, word_records, utterances, device,
    fps=PARAKEET_FPS, target_sr=WHISPER_SR, batch_size=32, checkpoint_dir=None,
) -> np.ndarray:
    model_name        = "parakeet-ctc-0.6b"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model             = ParakeetForCTC.from_pretrained(model_id, torch_dtype=torch.float32).to(device).eval()

    utt_to_words = _group_by_utt(word_records)
    utt_ids      = list(utt_to_words.keys())
    audio_key    = _resample_utterances(utterances, utt_ids, target_sr)

    checkpoint_path = checkpoint_dir / f"mcv_{model_name}_checkpoint.pkl" if checkpoint_dir else None
    start_utt, word_embeddings_list = _load_checkpoint(checkpoint_path, model_name)
    completed_words = {idx for idx, _ in word_embeddings_list}

    def encode(audio_arrays):
        inputs     = feature_extractor(audio_arrays, sampling_rate=target_sr,
                                       return_tensors="pt", padding="longest")
        input_key  = next(k for k in inputs.keys() if "mask" not in k)
        inp        = inputs[input_key].to(device, dtype=torch.float32)
        with torch.no_grad():
            if hasattr(model, "parakeet") and hasattr(model.parakeet, "encoder"):
                enc_out    = model.parakeet.encoder(inp, output_hidden_states=True)
                last_hidden = enc_out.last_hidden_state
            else:
                out         = model(inp, output_hidden_states=True)
                last_hidden = out.hidden_states[-1]
        return last_hidden.float().cpu().numpy()

    errors = _run_batch_audio(
        model_name, utt_ids, start_utt, batch_size,
        word_records, utterances, utt_to_words, completed_words,
        word_embeddings_list, encode, fps, checkpoint_path, audio_key,
    )
    del model
    release_vram(model_name)
    _remove_checkpoint(checkpoint_path)
    if errors:
        logger.warning(f"{model_name}: {errors} errors")
    return _sort_embeddings(word_embeddings_list, len(word_records))


# ---------------------------------------------------------------------------
# Mimi
# ---------------------------------------------------------------------------

def extract_mimi_word_embeddings(
    model_id, word_records, utterances, device,
    fps=MIMI_FPS, target_sr=MIMI_SR, batch_size=64, checkpoint_dir=None,
) -> np.ndarray:
    model_name        = "mimi"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model             = MimiModel.from_pretrained(model_id, torch_dtype=torch.float32).to(device).eval()

    utt_to_words = _group_by_utt(word_records)
    utt_ids      = list(utt_to_words.keys())
    audio_key    = _resample_utterances(utterances, utt_ids, target_sr)

    checkpoint_path = checkpoint_dir / f"mcv_{model_name}_checkpoint.pkl" if checkpoint_dir else None
    start_utt, word_embeddings_list = _load_checkpoint(checkpoint_path, model_name)
    completed_words = {idx for idx, _ in word_embeddings_list}

    def encode(audio_arrays):
        inputs = feature_extractor(raw_audio=audio_arrays, sampling_rate=target_sr,
                                   return_tensors="pt", padding=True)
        inp    = inputs["input_values"].to(device)
        with torch.no_grad():
            enc_out = model.encoder(inp)
        h = enc_out.float() if isinstance(enc_out, torch.Tensor) else enc_out.last_hidden_state.float()
        hidden_size = getattr(model.config, "hidden_size", None)
        if (h.dim() == 3 and hidden_size is not None
                and h.shape[1] == hidden_size and h.shape[2] != hidden_size):
            return h.cpu().numpy().transpose(0, 2, 1)
        return h.cpu().numpy()

    errors = _run_batch_audio(
        model_name, utt_ids, start_utt, batch_size,
        word_records, utterances, utt_to_words, completed_words,
        word_embeddings_list, encode, fps, checkpoint_path, audio_key,
    )
    del model
    release_vram(model_name)
    _remove_checkpoint(checkpoint_path)
    if errors:
        logger.warning(f"{model_name}: {errors} errors")
    return _sort_embeddings(word_embeddings_list, len(word_records))


# ---------------------------------------------------------------------------
# Text LM
# ---------------------------------------------------------------------------

def extract_lm_word_embeddings(
    model_name, model_id, word_records, device,
    batch_size=32, checkpoint_dir=None,
) -> np.ndarray:
    logger.info(f"Loading LM: {model_id}")
    model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                                     trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    utt_to_words = _group_by_utt(word_records)
    utt_ids      = list(utt_to_words.keys())
    N_utt        = len(utt_ids)

    checkpoint_path = checkpoint_dir / f"mcv_{model_name}_checkpoint.pkl" if checkpoint_dir else None
    start_utt, word_embeddings_list = _load_checkpoint(checkpoint_path, model_name)
    completed_words = {idx for idx, _ in word_embeddings_list}

    errors = 0
    pbar   = tqdm(range(start_utt, N_utt, batch_size), desc=model_name, unit="batch",
                  total=(N_utt - start_utt + batch_size - 1) // batch_size)

    for batch_start in pbar:
        batch_ids = utt_ids[batch_start : batch_start + batch_size]
        sentences = [word_records[utt_to_words[uid][0]]["sentence"] for uid in batch_ids]
        try:
            enc = tokenizer(sentences, return_tensors="pt", truncation=True,
                            max_length=MAX_TEXT_TOKENS, padding=True,
                            return_offsets_mapping=True)
            offset_mapping = enc.pop("offset_mapping").tolist()
            input_ids      = enc["input_ids"].to(device)
            attn_mask      = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            output_hidden_states=True)
            hidden_batch = out.hidden_states[-1].float().cpu().numpy()

            for b, utt_id in enumerate(batch_ids):
                hidden = hidden_batch[b]
                om     = offset_mapping[b]
                for word_idx in utt_to_words[utt_id]:
                    if word_idx in completed_words:
                        continue
                    rec    = word_records[word_idx]
                    char_s, char_e = rec["char_start"], rec["char_end"]
                    if char_s < 0:
                        errors += 1
                        continue
                    token_indices = [t for t, (ts, te) in enumerate(om)
                                     if ts < char_e and te > char_s and ts < te]
                    if not token_indices:
                        errors += 1
                        continue
                    word_embeddings_list.append((word_idx, hidden[token_indices].mean(axis=0)))
                    completed_words.add(word_idx)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _save_checkpoint(checkpoint_path, word_embeddings_list, batch_start)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch at utt {batch_start}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch at utt {batch_start}: {e}")

        next_utt = batch_start + batch_size
        if next_utt % (CHECKPOINT_EVERY * batch_size) < batch_size:
            _save_checkpoint(checkpoint_path, word_embeddings_list, next_utt)

    del model
    release_vram(model_name)
    _remove_checkpoint(checkpoint_path)
    if errors:
        logger.warning(f"{model_name}: {errors} errors")
    return _sort_embeddings(word_embeddings_list, len(word_records))


# ---------------------------------------------------------------------------
# Helpers  (identical to mls_word_level_analysis.py)
# ---------------------------------------------------------------------------

def _group_by_utt(word_records):
    groups = {}
    for idx, rec in enumerate(word_records):
        groups.setdefault(rec["utt_id"], []).append(idx)
    return groups


def _sort_embeddings(pairs, n_words):
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    if not pairs_sorted:
        raise RuntimeError("No word embeddings were collected.")
    D      = pairs_sorted[0][1].shape[0]
    result = np.zeros((n_words, D), dtype=np.float32)
    for idx, emb in pairs_sorted:
        result[idx] = emb
    return result


# ---------------------------------------------------------------------------
# Cross-Speaker Minibatch CKA with word-type mask
# ---------------------------------------------------------------------------

def _hsic1_batch(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC on pre-computed (masked) kernel matrices."""
    n = K.shape[0]
    assert n >= 4
    K = K.copy(); L = L.copy()
    np.fill_diagonal(K, 0.0); np.fill_diagonal(L, 0.0)
    KL    = K @ L
    ones  = np.ones(n)
    term1 = np.trace(KL)
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * (ones @ KL @ ones)
    return float((term1 + term2 - term3) / (n * (n - 3)))


def minibatch_cka(X, Y, word_ids, batch_size=MINIBATCH_SIZE, seed=MINIBATCH_SEED):
    """Minibatch linear CKA with word-type mask.

    Only same-word different-speaker pairs drive the HSIC score: within each
    minibatch the kernel matrices K and L are zeroed out for all (i, j) pairs
    where word_ids[i] != word_ids[j] before calling the unbiased HSIC estimator.
    """
    valid = (np.linalg.norm(X, axis=1) > 1e-10) & (np.linalg.norm(Y, axis=1) > 1e-10)
    X, Y  = X[valid], Y[valid]
    word_ids = word_ids[valid]
    N     = X.shape[0]
    rng   = np.random.default_rng(seed)
    indices = rng.permutation(N)
    hsic_xy, hsic_xx, hsic_yy = [], [], []

    for start in range(0, N - batch_size + 1, batch_size):
        idx  = indices[start : start + batch_size]
        Xb   = X[idx].astype(np.float64)
        Yb   = Y[idx].astype(np.float64)
        wids = word_ids[idx]

        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-10
        Yb /= np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-10

        K = Xb @ Xb.T
        L = Yb @ Yb.T

        # Word-type mask: zero out cross-word pairs
        mask = wids[:, None] == wids[None, :]
        K[~mask] = 0.0
        L[~mask] = 0.0

        hsic_xy.append(_hsic1_batch(K, L))
        hsic_xx.append(_hsic1_batch(K, K))
        hsic_yy.append(_hsic1_batch(L, L))

    if not hsic_xy:
        return 0.0, 0.0
    mean_xy = float(np.mean(hsic_xy))
    denom   = np.sqrt(max(float(np.mean(hsic_xx)), 0.0) * max(float(np.mean(hsic_yy)), 0.0))
    score   = float(mean_xy / denom) if denom > 1e-10 else 0.0
    per_batch_cka = [
        hxy / np.sqrt(max(hxx, 0.0) * max(hyy, 0.0))
        if np.sqrt(max(hxx, 0.0) * max(hyy, 0.0)) > 1e-10 else 0.0
        for hxy, hxx, hyy in zip(hsic_xy, hsic_xx, hsic_yy)
    ]
    k    = len(per_batch_cka)
    ci95 = 1.96 * float(np.std(per_batch_cka)) / np.sqrt(k) if k > 1 else 0.0
    return score, ci95


def run_cka_multi(embeddings, names, word_ids, batch_size, num_runs, base_seed=MINIBATCH_SEED):
    """Run pairwise CKA across num_runs shuffles, passing word_ids for masking."""
    n       = len(names)
    n_pairs = n * (n - 1) // 2
    scores_by_run = np.zeros((num_runs, n, n), dtype=np.float64)

    within_ci95 = np.zeros((n, n))
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        logger.info(f"CKA run {run_idx + 1}/{num_runs}  (seed={seed})")
        pair_bar = tqdm(total=n_pairs, desc=f"Run {run_idx+1}/{num_runs} CKA pairs", unit="pair")
        for i in range(n):
            for j in range(n):
                if i == j:
                    scores_by_run[run_idx, i, j] = 1.0
                    continue
                if j < i:
                    scores_by_run[run_idx, i, j] = scores_by_run[run_idx, j, i]
                    continue
                score, ci95 = minibatch_cka(embeddings[names[i]], embeddings[names[j]],
                                            word_ids=word_ids,
                                            batch_size=batch_size, seed=seed)
                scores_by_run[run_idx, i, j] = score
                if run_idx == 0:
                    within_ci95[i, j] = within_ci95[j, i] = ci95
                pair_bar.update(1)
        pair_bar.close()

    mean_matrix = scores_by_run.mean(axis=0)
    std_matrix  = scores_by_run.std(axis=0, ddof=1) if num_runs > 1 else np.zeros((n, n))
    ci95_matrix = 1.96 * std_matrix / np.sqrt(num_runs) if num_runs > 1 else within_ci95

    results_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{names[i]} vs {names[j]}"
            run_scores = scores_by_run[:, i, j].tolist()
            results_dict[key] = {
                "scores": run_scores,
                "mean":   float(mean_matrix[i, j]),
                "std":    float(std_matrix[i, j]),
                "ci95":   float(ci95_matrix[i, j]),
            }
            logger.info(f"  CKA({names[i]}, {names[j]})  "
                        f"mean={mean_matrix[i,j]:.4f}  std={std_matrix[i,j]:.4f}  "
                        f"95% CI ±{ci95_matrix[i,j]:.4f}")

    return mean_matrix, ci95_matrix, results_dict


# ---------------------------------------------------------------------------
# PCA  (identical to mls_word_level_analysis.py)
# ---------------------------------------------------------------------------

def pca_eigenvalues(X, n_components=PCA_COMPONENTS):
    from sklearn.utils.extmath import randomized_svd
    valid = np.linalg.norm(X, axis=1) > 1e-10
    Xv    = X[valid].astype(np.float64)
    Xc    = Xv - Xv.mean(axis=0, keepdims=True)
    n     = Xc.shape[0]
    if not np.isfinite(Xc).all():
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    _, s, _ = randomized_svd(
        Xc / np.sqrt(n - 1),
        n_components=min(n_components, min(Xc.shape) - 1),
        n_oversamples=10, n_iter=4, random_state=MINIBATCH_SEED,
    )
    eigs  = s ** 2
    eigs /= eigs.sum()
    return eigs


def effective_rank(eigenvalues):
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


# ---------------------------------------------------------------------------
# Plots  (MLS logic, filenames/titles updated to MCV Cross-Speaker)
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _draw_cka_heatmap_axes(ax, cka_matrix, names, ci95_matrix=None,
                            title="MCV Cross-Speaker Word-Level Pairwise Minibatch Linear CKA"):
    cmap = LinearSegmentedColormap.from_list("cka", ["#FFFFFF", "#A5D6A7", "#2E7D32"], N=256)
    im   = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA  (0 = no similarity,  1 = identical)", fontsize=9)

    n = len(names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val        = cka_matrix[i, j]
            text_color = "black" if val < 0.7 else "white"
            if i == j:
                label = f"{val:.3f}"
            elif ci95_matrix is not None:
                label = f"{val:.3f}\n±{ci95_matrix[i,j]:.3f}"
            else:
                label = f"{val:.3f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color=text_color)

    audio_indices = [i for i, name in enumerate(names)
                     if MODELS.get(name, {}).get("modality") in AUDIO_MODALITIES]
    if audio_indices:
        boundary = max(audio_indices) + 0.5
        ax.axhline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.axvline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)


def plot_cka_heatmap(cka_matrix, names, plots_dir, ci95_matrix=None,
                     filename="mcv_word_cka_heatmap.png"):
    _apply_style()
    fig, ax = plt.subplots(figsize=(max(8, len(names)), max(6.5, len(names) * 0.9)))
    _draw_cka_heatmap_axes(ax, cka_matrix, names, ci95_matrix=ci95_matrix)
    plt.tight_layout()
    path = plots_dir / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_cka_multi_run(mean_matrix, ci95_matrix, per_run_matrices, names, plots_dir, num_runs):
    _apply_style()
    n         = len(names)
    cell_size = max(0.9, min(1.1, 9.0 / n))

    if num_runs <= 6:
        fig = plt.figure(figsize=(max(10, n * cell_size * num_runs * 0.5 + 2),
                                  max(8, n * cell_size * 2 + 2)))
        ax_mean = fig.add_subplot(2, num_runs, (1, num_runs))
        _draw_cka_heatmap_axes(ax_mean, mean_matrix, names, ci95_matrix=ci95_matrix,
                               title=f"MCV Cross-Speaker Mean CKA across {num_runs} shuffles\n(±95% CI across runs)")
        for r in range(num_runs):
            ax_r = fig.add_subplot(2, num_runs, num_runs + r + 1)
            _draw_cka_heatmap_axes(ax_r, per_run_matrices[r], names,
                                   title=f"Run {r + 1}  (seed={MINIBATCH_SEED + r})")
    else:
        fig, ax_mean = plt.subplots(figsize=(max(8, n * cell_size), max(6.5, n * cell_size * 0.9)))
        _draw_cka_heatmap_axes(ax_mean, mean_matrix, names, ci95_matrix=ci95_matrix,
                               title=f"MCV Cross-Speaker Mean CKA across {num_runs} shuffles\n(±95% CI across runs)")

    plt.tight_layout()
    path = plots_dir / "mcv_word_cka_heatmap_multi_run.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_cka_variability_scatter(results_dict, names, plots_dir):
    _apply_style()
    fig, ax   = plt.subplots(figsize=(7, 5))
    color_map = {"audio-audio": "#0288D1", "text-text": "#FB8C00", "cross-modal": "#2E7D32"}

    for key, vals in results_dict.items():
        a, b     = key.split(" vs ", 1)
        mod_a    = MODELS.get(a, {}).get("modality", "")
        mod_b    = MODELS.get(b, {}).get("modality", "")
        is_aud_a = mod_a in AUDIO_MODALITIES
        is_aud_b = mod_b in AUDIO_MODALITIES
        cat = ("audio-audio" if is_aud_a and is_aud_b
               else "text-text" if not is_aud_a and not is_aud_b
               else "cross-modal")
        ax.scatter(vals["mean"], vals["std"], color=color_map[cat], alpha=0.7, s=50, zorder=3)

    for cat, color in color_map.items():
        ax.scatter([], [], color=color, label=cat, s=50)
    ax.legend(fontsize=9)
    ax.set_xlabel("Mean CKA across runs", fontsize=11)
    ax.set_ylabel("Std of CKA across runs", fontsize=11)
    ax.set_title("MCV Cross-Speaker CKA Estimate Stability", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = plots_dir / "mcv_word_cka_variability_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_eigenspectra_overlay(eigenvalues, plots_dir, max_pc=PCA_PLOT_MAX_PC):
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, eigs_full in eigenvalues.items():
        eigs  = eigs_full[:max_pc]
        color = MODEL_COLORS.get(name, "#555555")
        ax.plot(np.arange(1, len(eigs) + 1), eigs, color=color, linewidth=2,
                label=name, alpha=0.85)
    ax.set_xlabel(f"Principal Component (1–{max_pc})", fontsize=11)
    ax.set_ylabel("Fraction of Variance", fontsize=11)
    ax.set_yscale("log"); ax.set_xlim(1, max_pc)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"MCV Cross-Speaker Word-Level PCA Eigenvalue Spectra  ·  first {max_pc} PCs",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = plots_dir / "mcv_word_eigenspectra_overlay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_effective_rank_bar(eigenvalues, plots_dir):
    _apply_style()
    names  = list(eigenvalues.keys())
    ranks  = [effective_rank(eigenvalues[n]) for n in names]
    colors = [MODEL_COLORS.get(n, "#555555") for n in names]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    bars    = ax.bar(range(len(names)), ranks, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("MCV Cross-Speaker Word-Level Effective Rank per Model", fontsize=12, fontweight="bold")
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{rank:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = plots_dir / "mcv_word_effective_rank.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Save tables  (filenames prefixed with mcv_)
# ---------------------------------------------------------------------------

def save_summary_tables(mean_matrix, ci95_matrix, results_dict, names,
                        eigenvalues, embeddings, data_dir, per_run_matrices=None):
    path1 = data_dir / "mcv_word_cka_matrix.csv"
    with open(path1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, row_name in enumerate(names):
            w.writerow([row_name] + [f"{mean_matrix[i,j]:.6f}" for j in range(len(names))])
    logger.info(f"  Saved → {path1}")

    if per_run_matrices:
        for r, mat in enumerate(per_run_matrices):
            path_r = data_dir / f"mcv_word_cka_matrix_run{r+1}.csv"
            with open(path_r, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([""] + names)
                for i, row_name in enumerate(names):
                    w.writerow([row_name] + [f"{mat[i,j]:.6f}" for j in range(len(names))])

    path2 = data_dir / "mcv_word_cka_stats.csv"
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        num_runs = len(next(iter(results_dict.values()))["scores"]) if results_dict else 0
        w.writerow(["pair", "mean_cka", "std_across_runs", "ci95_across_runs"]
                   + [f"score_run{r+1}" for r in range(num_runs)])
        for pair, vals in results_dict.items():
            w.writerow([pair, f"{vals['mean']:.6f}", f"{vals['std']:.6f}", f"{vals['ci95']:.6f}"]
                       + [f"{s:.6f}" for s in vals["scores"]])
    logger.info(f"  Saved → {path2}")

    path3 = data_dir / "mcv_word_pca_summary.csv"
    with open(path3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "arch", "embedding_dim",
                    "effective_rank", "var_pc1", "var_top5", "var_top10"])
        for name, eigs in eigenvalues.items():
            dim = embeddings[name].shape[1] if name in embeddings else "—"
            w.writerow([name, MODELS[name]["params"], MODELS[name]["arch"], dim,
                        f"{effective_rank(eigs):.2f}", f"{eigs[0]:.4f}",
                        f"{eigs[:5].sum():.4f}", f"{eigs[:10].sum():.4f}"])
    logger.info(f"  Saved → {path3}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Mozilla Common Voice English word-level cross-speaker CKA "
                    "(download → CTC align → embed → analyse)"
    )
    p.add_argument("--root_dir", default=".", type=Path,
                   help="Root directory for MCVData/, MCVPlots/, logs/")
    p.add_argument("--mcv_dir", default=None, type=Path,
                   help="Path to an existing MCV sample directory (wavs/ + transcripts.json). "
                        "If omitted, defaults to <root_dir>/MCVData/mcv_sample and downloads.")
    p.add_argument("--textgrid_dir", default=None, type=Path,
                   help="Path to an existing TextGrid directory. "
                        "If omitted, defaults to <root_dir>/MCVData/textgrids and runs CTC alignment.")
    p.add_argument("--n_words", default=250_000, type=int,
                   help="Target number of word tokens to sample (default: 250000). "
                        "Ignored if --n_utterances is set.")
    p.add_argument("--n_utterances", default=None, type=int,
                   help="Sample exactly this many utterances instead of targeting a word count.")
    p.add_argument("--sample_seed", default=42, type=int,
                   help="Random seed for sampling")
    p.add_argument("--min_speakers", default=50, type=int,
                   help="Minimum number of distinct speakers required per sentence (default: 50)")
    p.add_argument("--mcv_source_dir", required=True, type=Path,
                   help="Path to local MCV English directory containing validated.tsv and clips/")
    p.add_argument("--tsv_file", default="validated.tsv",
                   help="TSV filename inside mcv_source_dir (default: validated.tsv)")
    p.add_argument("--dry_run", action="store_true",
                   help="Run sampling and print stats without converting any audio")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip download stage (mcv_dir must already exist)")
    p.add_argument("--skip_mfa", action="store_true",
                   help="Skip alignment stage (textgrid_dir must already exist)")
    p.add_argument("--skip_extraction", action="store_true",
                   help="Skip embedding extraction (use cached .pkl files only)")
    p.add_argument("--max_words", default=None, type=int,
                   help="Cap word records (for testing)")
    p.add_argument("--batch_size", default=2048, type=int,
                   help="Minibatch size for CKA")
    p.add_argument("--num_cka_runs", default=1, type=int,
                   help="Number of CKA shuffle runs for variability estimation")
    p.add_argument("--cka_base_seed", default=MINIBATCH_SEED, type=int)
    p.add_argument("--ctc_batch_size", default=8, type=int,
                   help="Number of utterances between GPU cache clears during CTC alignment "
                        "(default: 8; raise if you have lots of VRAM)")
    p.add_argument("--whisper_batch_size",     default=64,  type=int)
    p.add_argument("--whisper_dec_batch_size", default=32,  type=int)
    p.add_argument("--parakeet_batch_size",    default=32,  type=int)
    p.add_argument("--mimi_batch_size",        default=64,  type=int)
    p.add_argument("--lm_batch_size",          default=32,  type=int)
    p.add_argument("--pca_components",         default=50,  type=int)
    p.add_argument("--pca_plot_max_pc",        default=10,  type=int)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args      = parse_args()
    root      = Path(args.root_dir)
    data_dir  = root / "MCVData"
    plots_dir = root / "MCVPlots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(root / "logs")
    logger.info("=" * 60)
    logger.info("MCV Cross-Speaker Word-Level Representation Analysis — run started")
    logger.info(f"Log: {log_path}")
    logger.info(f"Args: {vars(args)}")
    logger.info("=" * 60)

    device = get_device()

    mcv_dir      = args.mcv_dir      or (data_dir / "mcv_sample")
    textgrid_dir = args.textgrid_dir or (data_dir / "textgrids")

    # ------------------------------------------------------------------
    # Stage 0: Download
    # ------------------------------------------------------------------
    if not args.skip_download:
        with timer("Download MCV sample"):
            download_mcv_sample(
                mcv_source_dir=args.mcv_source_dir,
                mcv_dir=mcv_dir,
                n_utterances=args.n_utterances,
                n_words=args.n_words,
                seed=args.sample_seed,
                min_speakers=args.min_speakers,
                tsv_file=args.tsv_file,
                dry_run=args.dry_run,
            )
    else:
        logger.info(f"--skip_download: using existing MCV sample at {mcv_dir}")

    if args.dry_run:
        logger.info("--dry_run: exiting after sampling stage")
        return

    # ------------------------------------------------------------------
    # Stage 1: CTC alignment
    # ------------------------------------------------------------------
    if not args.skip_mfa:
        with timer("CTC forced alignment"):
            run_ctc_alignment(mcv_dir, textgrid_dir, batch_size=args.ctc_batch_size)
    else:
        logger.info(f"--skip_mfa: using existing TextGrids at {textgrid_dir}")

    # ------------------------------------------------------------------
    # Stage 2: Load utterances + build word records
    # ------------------------------------------------------------------
    with timer("Load MCV utterances"):
        utterances = load_mcv_utterances(mcv_dir)

    cache_suffix      = f"_max{args.max_words}" if args.max_words else ""
    word_records_path = data_dir / f"mcv_word_records{cache_suffix}.json"

    if word_records_path.exists():
        logger.info(f"Loading cached word records from {word_records_path}")
        with open(word_records_path) as f:
            word_records = json.load(f)
        logger.info(f"Loaded {len(word_records):,} word records")
    else:
        with timer("Build word records"):
            word_records = build_word_records(
                utterances, textgrid_dir, max_words=args.max_words,
            )
        with open(word_records_path, "w") as f:
            json.dump(word_records, f)
        logger.info(f"Cached word records → {word_records_path}")

    N = len(word_records)
    logger.info(f"Total word tokens: {N:,}")

    # Build word-type IDs for cross-speaker CKA masking
    word_ids = build_word_type_ids(word_records)
    logger.info(f"Word type vocabulary size: {len(set(word_ids.tolist())):,}")

    # ------------------------------------------------------------------
    # Stage 3: Extract / load embeddings
    # ------------------------------------------------------------------
    embeddings = {}
    for model_name, cfg in MODELS.items():
        cache_path = data_dir / f"mcv_word_embeddings_{model_name}.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached embeddings: {model_name}")
            with open(cache_path, "rb") as f:
                embeddings[model_name] = pickle.load(f)
            logger.info(f"  Shape: {embeddings[model_name].shape}")
            continue

        if args.skip_extraction:
            logger.warning(f"--skip_extraction but no cache for {model_name}, skipping")
            continue

        logger.info("=" * 60)
        logger.info(f"Extracting: {model_name}  [{cfg['params']} | {cfg['arch']}]")
        logger.info("=" * 60)

        with timer(f"Extract {model_name}"):
            try:
                modality = cfg["modality"]
                if modality == "audio-whisper-enc":
                    emb = extract_whisper_enc_word_embeddings(
                        model_name, cfg["hf_id"], word_records, utterances, device,
                        fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.whisper_batch_size, checkpoint_dir=data_dir,
                    )
                elif modality == "audio-whisper-dec":
                    emb = extract_whisper_dec_word_embeddings(
                        model_name, cfg["hf_id"], word_records, utterances, device,
                        target_sr=cfg["target_sr"],
                        batch_size=args.whisper_dec_batch_size, checkpoint_dir=data_dir,
                    )
                elif modality == "audio-parakeet":
                    emb = extract_parakeet_word_embeddings(
                        cfg["hf_id"], word_records, utterances, device,
                        fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.parakeet_batch_size, checkpoint_dir=data_dir,
                    )
                elif modality == "audio-mimi":
                    emb = extract_mimi_word_embeddings(
                        cfg["hf_id"], word_records, utterances, device,
                        fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.mimi_batch_size, checkpoint_dir=data_dir,
                    )
                else:
                    emb = extract_lm_word_embeddings(
                        model_name, cfg["hf_id"], word_records, device,
                        batch_size=args.lm_batch_size, checkpoint_dir=data_dir,
                    )
            except Exception as e:
                logger.error(f"Extraction failed for {model_name}: {e}. Skipping.")
                release_vram(f"after {model_name} failure")
                continue

        n_bad = (~np.isfinite(emb)).sum()
        if n_bad:
            logger.warning(f"{model_name}: {n_bad} non-finite values")

        embeddings[model_name] = emb
        with open(cache_path, "wb") as f:
            pickle.dump(emb, f)
        logger.info(f"Cached → {cache_path}  ({cache_path.stat().st_size / 1024**2:.1f} MB)")

    names = list(embeddings.keys())
    logger.info(f"Models with embeddings: {names}")

    # ------------------------------------------------------------------
    # Align embeddings + filter word_ids in parallel
    # ------------------------------------------------------------------
    if embeddings:
        valid_mask   = np.ones(N, dtype=bool)
        for emb in embeddings.values():
            valid_mask &= np.linalg.norm(emb, axis=1) > 1e-10
        n_valid      = int(valid_mask.sum())
        n_dropped    = N - n_valid
        frac_dropped = n_dropped / N
        logger.info(f"Word alignment: {n_valid:,} / {N:,} valid ({n_dropped:,} dropped, "
                    f"{frac_dropped:.2%})")
        if frac_dropped > 0.02:
            raise RuntimeError(
                f"{frac_dropped:.2%} words dropped during alignment — exceeds 2% threshold.\n"
                + "\n".join(f"  {n}: {int((np.linalg.norm(e, axis=1) <= 1e-10).sum()):,} zeros"
                            for n, e in embeddings.items())
            )
        if n_dropped > 0:
            for name in list(embeddings.keys()):
                embeddings[name] = embeddings[name][valid_mask]
            word_ids = word_ids[valid_mask]
            N = n_valid

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    logger.info("=" * 60); logger.info("PCA eigenvalue analysis"); logger.info("=" * 60)
    eigenvalues = {}
    with timer("PCA"):
        for name, X in embeddings.items():
            eigs            = pca_eigenvalues(X, n_components=args.pca_components)
            eigenvalues[name] = eigs
            logger.info(f"  {name:<22}  dim={X.shape[1]:>5}  eff_rank={effective_rank(eigs):.1f}")

    with timer("PCA plots"):
        plot_eigenspectra_overlay(eigenvalues, plots_dir, max_pc=args.pca_plot_max_pc)
        plot_effective_rank_bar(eigenvalues, plots_dir)

    # ------------------------------------------------------------------
    # Cross-Speaker CKA
    # ------------------------------------------------------------------
    num_runs = max(1, args.num_cka_runs)
    logger.info("=" * 60)
    logger.info(f"Pairwise MCV cross-speaker word-level CKA  "
                f"(batch_size={args.batch_size}, N={N:,}, num_runs={num_runs})")
    logger.info("=" * 60)

    with timer(f"Pairwise CKA ({num_runs} run(s))"):
        mean_matrix, ci95_matrix, cka_results = run_cka_multi(
            embeddings, names, word_ids=word_ids,
            batch_size=args.batch_size,
            num_runs=num_runs, base_seed=args.cka_base_seed,
        )

    n = len(names)
    per_run_matrices = []
    for r in range(num_runs):
        mat = np.zeros((n, n))
        for i in range(n):
            mat[i, i] = 1.0
            for j in range(i + 1, n):
                key = f"{names[i]} vs {names[j]}"
                mat[i, j] = mat[j, i] = cka_results[key]["scores"][r]
        per_run_matrices.append(mat)

    with timer("CKA plots"):
        if num_runs == 1:
            plot_cka_heatmap(mean_matrix, names, plots_dir,
                             ci95_matrix=ci95_matrix, filename="mcv_word_cka_heatmap.png")
        else:
            plot_cka_multi_run(mean_matrix, ci95_matrix, per_run_matrices,
                               names, plots_dir, num_runs)
            plot_cka_heatmap(mean_matrix, names, plots_dir, ci95_matrix=ci95_matrix,
                             filename="mcv_word_cka_heatmap.png")
            plot_cka_variability_scatter(cka_results, names, plots_dir)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    with timer("Save tables"):
        save_summary_tables(mean_matrix, ci95_matrix, cka_results, names,
                            eigenvalues, embeddings, data_dir,
                            per_run_matrices=per_run_matrices if num_runs > 1 else None)

    results_json = {
        "dataset":          "Mozilla Common Voice English",
        "n_utterances":     args.n_utterances,
        "n_words":          N,
        "num_cka_runs":     num_runs,
        "cka_base_seed":    args.cka_base_seed,
        "models":           {k: {kk: vv for kk, vv in v.items()
                                 if kk not in ("fps", "target_sr")} for k, v in MODELS.items()},
        "embedding_shapes": {k: list(v.shape) for k, v in embeddings.items()},
        "cka_results":      cka_results,
        "mean_cka_matrix":  mean_matrix.tolist(),
        "ci95_cka_matrix":  ci95_matrix.tolist(),
        "eigenvalue_spectra": {k: v.tolist() for k, v in eigenvalues.items()},
        "effective_ranks":  {k: effective_rank(v) for k, v in eigenvalues.items()},
    }
    json_path = data_dir / "mcv_word_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results JSON → {json_path}")
    logger.info("Run complete.")


if __name__ == "__main__":
    main()
