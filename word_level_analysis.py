#!/usr/bin/env python3
"""
Word-Level Representation Analysis: Cross-Modal Minibatch CKA & PCA

Extends the utterance-level analysis (representation_analysis.py) to
word-level embeddings using LJSpeech audio + TextGrid forced alignments.

For each word token in context:
  Audio models: run the full utterance through the encoder, then slice
                output frames by word boundary timestamps and mean-pool
                over those frames.
  Text models:  tokenize the full sentence (for context), run through the
                model, find the tokens corresponding to the target word
                via offset mapping, and mean-pool those token hidden states.

This tests whether cross-modal similarity holds at the word level, and
lets us examine which word types (content vs function, frequent vs rare)
drive or inhibit cross-modal alignment.

Data:
  LJSpeech audio  — downloaded automatically from keithito.com, or pass --ljspeech_dir
  TextGrids       — one .TextGrid file per utterance, named LJ{set}-{id}.TextGrid
                    e.g. from https://www.kaggle.com/datasets/b09901026skho/ljspeech-textgrid

Usage:
  python word_level_analysis.py                              # fully automatic (downloads everything)
  python word_level_analysis.py --ljspeech_dir /path/to/LJSpeech-1.1
  python word_level_analysis.py --ljspeech_dir /path/to/LJSpeech-1.1 --max_words 50000

Auto-download:
  LJSpeech:  if --ljspeech_dir is omitted, downloads LJSpeech-1.1 (~2.6 GB) from
             https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
  TextGrids: if --textgrid_dir is omitted, downloads from Kaggle. Requires:
               pip install kaggle
               # Place ~/.kaggle/kaggle.json with your API token
"""

import argparse
import csv
import tarfile
import urllib.request
import gc
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
    AutoModel,
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

LJSPEECH_SR      = 22050    # LJSpeech native sample rate
WHISPER_SR       = 16000    # Whisper / Parakeet target sample rate
MIMI_SR          = 24000    # Mimi target sample rate

# Output frame rates after model-specific downsampling
WHISPER_ENC_FPS  = 50.0     # Whisper encoder: 10ms mel hop × 2x conv = 20ms/frame
PARAKEET_FPS     = 12.5     # FastConformer: 10ms shift × 8x subsampling = 80ms/frame
MIMI_FPS         = 12.5     # Mimi encoder: 80ms/frame at 24kHz

MIN_WORD_DUR     = 0.05     # seconds — skip words shorter than this (< 1 Whisper frame)
MAX_WORD_DUR     = 3.0      # seconds — skip implausibly long words
MIN_WORD_ALPHA   = 2        # minimum alphabetic characters in word

CHECKPOINT_EVERY = 200      # save checkpoint every N batches processed
MINIBATCH_SIZE   = 2048
MINIBATCH_SEED   = 42
MAX_TEXT_TOKENS  = 256
PCA_COMPONENTS   = 50
PCA_PLOT_MAX_PC  = 10

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Decoder models (Whisper decoder, Mimi decoder, Voxtral) are excluded because
# word-level frame slicing is not well-defined for autoregressive decoders —
# decoder hidden states encode what the model predicts next, not what it heard.

MODELS = {
    # ── Audio encoders ────────────────────────────────────────────────────
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
    # ── Text LLMs ─────────────────────────────────────────────────────────
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

AUDIO_MODALITIES = {"audio-whisper-enc", "audio-parakeet", "audio-mimi"}

MODEL_COLORS = {
    "whisper-base-enc":    "#B3E5FC",
    "whisper-small-enc":   "#4FC3F7",
    "whisper-medium-enc":  "#0288D1",
    "whisper-large-enc":   "#01579B",
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

logger = logging.getLogger("word_level")


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"word_run_{timestamp}.log"
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
# LJSpeech auto-download
# ---------------------------------------------------------------------------

LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def ensure_ljspeech(ljspeech_dir: Path) -> Path:
    """Ensure LJSpeech-1.1 is present, downloading and extracting if needed.

    Returns the path to the LJSpeech-1.1 root (contains metadata.csv and wavs/).
    """
    def _has_wavs(d: Path) -> bool:
        return (d / "metadata.csv").exists() and (d / "wavs").is_dir() and any((d / "wavs").glob("*.wav"))

    # If directory already has metadata.csv and a populated wavs/, we're done
    if _has_wavs(ljspeech_dir):
        n_wavs = sum(1 for _ in (ljspeech_dir / "wavs").glob("*.wav"))
        logger.info(f"Found LJSpeech at {ljspeech_dir} ({n_wavs:,} wav files)")
        return ljspeech_dir

    # Check one level up in case the tarball extracted to a subdirectory
    candidate = ljspeech_dir / "LJSpeech-1.1"
    if _has_wavs(candidate):
        logger.info(f"Found LJSpeech at {candidate}")
        return candidate

    ljspeech_dir.mkdir(parents=True, exist_ok=True)
    tarball = ljspeech_dir / "LJSpeech-1.1.tar.bz2"

    if not tarball.exists():
        logger.info(f"Downloading LJSpeech-1.1 (~2.6 GB) from {LJSPEECH_URL} ...")
        try:
            def _progress(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100.0, block_num * block_size / total_size * 100)
                    if block_num % 500 == 0:
                        logger.info(f"  Download progress: {pct:.1f}%")
            urllib.request.urlretrieve(LJSPEECH_URL, tarball, reporthook=_progress)
        except Exception as e:
            raise RuntimeError(
                f"LJSpeech download failed: {e}\n"
                f"Download manually from {LJSPEECH_URL}\n"
                f"and extract to {ljspeech_dir}, then pass --ljspeech_dir to the script."
            )

    logger.info(f"Extracting {tarball} ...")
    try:
        with tarfile.open(tarball, "r:bz2") as tf:
            try:
                tf.extractall(ljspeech_dir, filter="data")  # Python 3.12+ safe extraction
            except TypeError:
                tf.extractall(ljspeech_dir)                 # Python < 3.12 fallback
    except Exception as e:
        raise RuntimeError(f"Failed to extract {tarball}: {e}")

    tarball.unlink()   # remove the tarball to save ~2.6 GB

    # After extraction, LJSpeech-1.1/ should be a subdirectory
    if (candidate / "metadata.csv").exists():
        logger.info(f"LJSpeech ready at {candidate}")
        return candidate

    raise RuntimeError(
        f"Extraction completed but metadata.csv not found under {ljspeech_dir}.\n"
        f"Check the contents of {ljspeech_dir} and pass the correct path with --ljspeech_dir."
    )


# ---------------------------------------------------------------------------
# TextGrid auto-download
# ---------------------------------------------------------------------------

KAGGLE_DATASET = "b09901026skho/ljspeech-textgrid"
TEXTGRID_PRESENCE_THRESHOLD = 100   # if this many .TextGrid files exist, skip download


def ensure_textgrids(textgrid_dir: Path) -> Path:
    """Ensure TextGrid files are present in textgrid_dir, downloading from Kaggle if needed.

    Returns the (possibly updated) path to the directory that actually contains
    the .TextGrid files — Kaggle may unzip into a subdirectory.

    Raises RuntimeError with a helpful message if download fails and no files exist.
    """
    existing = list(textgrid_dir.rglob("*.TextGrid")) if textgrid_dir.exists() else []
    if len(existing) >= TEXTGRID_PRESENCE_THRESHOLD:
        # Find the common directory containing the files (handles subdirectory cases)
        dirs = set(f.parent for f in existing)
        best = min(dirs, key=lambda d: len(d.parts))
        if best != textgrid_dir:
            logger.info(f"Found {len(existing):,} TextGrid files in {best}")
        else:
            logger.info(f"Found {len(existing):,} TextGrid files in {textgrid_dir}")
        return best

    textgrid_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"No TextGrid files found in {textgrid_dir}. Attempting Kaggle download...")

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The 'kaggle' CLI is not available. Install with:\n"
            "  pip install kaggle\n"
            "Then set up your API token:\n"
            "  1. Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "  2. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json\n"
            f"Or download manually: https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"and extract to: {textgrid_dir}"
        )

    logger.info(f"Downloading {KAGGLE_DATASET} → {textgrid_dir} (this may take a while)...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(textgrid_dir), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed:\n{result.stderr}\n"
            "Make sure your API token is set up:\n"
            "  1. Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "  2. Place kaggle.json at ~/.kaggle/kaggle.json\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json\n"
            f"Or download manually: https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"and extract to: {textgrid_dir}"
        )

    # After unzip, files may be in a subdirectory — find where they actually are
    all_tg = list(textgrid_dir.rglob("*.TextGrid"))
    if not all_tg:
        raise RuntimeError(
            f"Download completed but no .TextGrid files found under {textgrid_dir}.\n"
            "The dataset may use a different file extension or naming convention.\n"
            f"Check the contents of {textgrid_dir} and pass the correct path with --textgrid_dir."
        )

    dirs = set(f.parent for f in all_tg)
    actual_dir = min(dirs, key=lambda d: len(d.parts))
    logger.info(f"Download complete — {len(all_tg):,} TextGrid files in {actual_dir}")
    return actual_dir


# ---------------------------------------------------------------------------
# TextGrid parser
# ---------------------------------------------------------------------------

def parse_textgrid(path: Path) -> list[dict]:
    """Parse a Praat TextGrid file and return word intervals.

    Returns list of dicts: {"word": str, "start": float, "end": float}
    Silence intervals (empty text, "sp", "sil", "<eps>") are skipped.
    Handles both short (compact) and long (verbose) TextGrid formats.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    words = []

    # Try to find the words/word tier block
    # Look for: name = "words" or name = "word" (case-insensitive)
    for tier_name in ('"words"', '"word"', '"Words"', '"Word"'):
        pattern = re.compile(
            rf'name = {re.escape(tier_name)}.*?(?=\bitem\s*\[|\Z)',
            re.DOTALL,
        )
        m = pattern.search(text)
        if m:
            tier_text = m.group(0)
            break
    else:
        logger.warning(f"No words tier found in {path.name}")
        return words

    # Extract all intervals from the tier block
    interval_pattern = re.compile(
        r'xmin\s*=\s*([\d.eE+\-]+)\s*\n\s*xmax\s*=\s*([\d.eE+\-]+)\s*\n\s*(?:text|intervals\s*\[\d+\].*?text)\s*=\s*"([^"]*)"',
        re.DOTALL,
    )
    for m in interval_pattern.finditer(tier_text):
        word = m.group(3).strip()
        start = float(m.group(1))
        end   = float(m.group(2))
        # Skip silence/pause markers
        if not word or word.lower() in ("sp", "sil", "<eps>", "spn", "{lg}", "{ns}"):
            continue
        words.append({"word": word, "start": start, "end": end})

    return words


# ---------------------------------------------------------------------------
# Data loading: LJSpeech + TextGrids
# ---------------------------------------------------------------------------

def load_ljspeech(ljspeech_dir: str) -> dict:
    """Load LJSpeech from a local directory. Returns dict: utt_id → {audio, sr, text}.

    Expects the standard LJSpeech-1.1 layout:
      <ljspeech_dir>/metadata.csv          (id|text|normalized_text, pipe-separated)
      <ljspeech_dir>/wavs/LJ001-0001.wav   (22050 Hz mono WAV)
    """
    ljspeech_dir = Path(ljspeech_dir)
    metadata_path = ljspeech_dir / "metadata.csv"
    wavs_dir      = ljspeech_dir / "wavs"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found in {ljspeech_dir}. "
            "Pass the path to the LJSpeech-1.1 root directory via --ljspeech_dir."
        )

    logger.info(f"Loading LJSpeech from {ljspeech_dir}...")
    utterances = {}
    with open(metadata_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading LJSpeech", unit="utt"):
            line = line.rstrip("\n")
            parts = line.split("|")
            if len(parts) < 2:
                continue
            utt_id   = parts[0]
            # metadata.csv columns: id | text | normalized_text (normalized may be absent)
            text = parts[2] if len(parts) >= 3 and parts[2] else parts[1]

            wav_path = wavs_dir / f"{utt_id}.wav"
            if not wav_path.exists():
                continue

            waveform, sr = torchaudio.load(str(wav_path))
            audio = waveform.squeeze(0).numpy().astype(np.float32)

            utterances[utt_id] = {"audio": audio, "sr": int(sr), "text": text.lower()}

    logger.info(f"Loaded {len(utterances)} LJSpeech utterances")
    return utterances


def build_word_records(
    utterances: dict,
    textgrid_dir: Path,
    min_dur: float = MIN_WORD_DUR,
    max_dur: float = MAX_WORD_DUR,
    max_words: int = None,
) -> list[dict]:
    """Combine LJSpeech utterances with TextGrid word boundaries.

    Returns a list of word records:
      {utt_id, word, start, end, sentence, char_start, char_end}

    char_start / char_end are character positions within sentence (lowercase),
    used by text models to find the correct tokens via offset mapping.
    """
    records = []
    missing_tg = 0

    for utt_id, utt in tqdm(utterances.items(), desc="Building word records", unit="utt"):
        tg_path = textgrid_dir / f"{utt_id}.TextGrid"
        if not tg_path.exists():
            missing_tg += 1
            continue

        word_intervals = parse_textgrid(tg_path)
        if not word_intervals:
            continue

        sentence = utt["text"].lower()  # LJSpeech normalized_text is uppercase; lowercase for matching

        # Track how many times each word has appeared so far in this utterance
        # so that _find_word_in_sentence finds the correct occurrence (not always the first).
        word_occurrence_count: dict[str, int] = {}

        for wi in word_intervals:
            word = wi["word"].lower()
            dur  = wi["end"] - wi["start"]

            # Duration filter
            if dur < min_dur or dur > max_dur:
                continue

            # Must contain enough alphabetic characters
            if sum(c.isalpha() for c in word) < MIN_WORD_ALPHA:
                continue

            # Find the Nth occurrence of this word in the sentence, where N is
            # how many times it has already appeared in this utterance's TextGrid.
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
            })

            if max_words and len(records) >= max_words:
                logger.info(f"Reached max_words={max_words}, stopping early")
                return records

    if missing_tg > 0:
        logger.warning(f"{missing_tg} utterances had no TextGrid file")
    logger.info(f"Built {len(records):,} word records from {len(utterances):,} utterances")
    return records


def _find_word_in_sentence(word: str, sentence: str, occurrence: int = 0) -> tuple[int, int]:
    """Return (char_start, char_end) of the nth occurrence (0-indexed) of word in sentence.

    Uses regex word boundaries to avoid matching substrings (e.g. "the" inside "there").
    Falls back to punctuation-stripped form if no boundary match is found.
    Returns (-1, -1) if the word is not found.
    """
    def _nth_match(pattern: str, text: str, n: int) -> tuple[int, int]:
        matches = list(re.finditer(pattern, text))
        if n < len(matches):
            m = matches[n]
            return m.start(), m.end()
        # Occurrence n not found; fall back to last available match
        if matches:
            m = matches[-1]
            return m.start(), m.end()
        return -1, -1

    # Try word-boundary match
    start, end = _nth_match(r'\b' + re.escape(word) + r'\b', sentence, occurrence)
    if start >= 0:
        return start, end

    # Try stripping punctuation from the word (e.g. "it's" → "its")
    clean = re.sub(r"[^a-z']", "", word)
    if clean and clean != word:
        start, end = _nth_match(r'\b' + re.escape(clean) + r'\b', sentence, occurrence)
        if start >= 0:
            return start, end

    return -1, -1


# ---------------------------------------------------------------------------
# Checkpoint helpers
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
        logger.debug(f"Checkpoint removed: {path.name}")


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample audio array from src_sr to dst_sr using torchaudio."""
    if src_sr == dst_sr:
        return audio
    t = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    t = torchaudio.functional.resample(t, orig_freq=src_sr, new_freq=dst_sr)
    return t.squeeze(0).numpy()


def _resample_utterances(utterances: dict, utt_ids: list, target_sr: int) -> str:
    """Pre-cache resampled audio for the given utterance IDs and sample rate.

    Audio is stored under the key f"audio_{target_sr}" so that different models
    with different target sample rates don't overwrite each other's cached arrays.
    Returns the cache key so callers can pass it directly to _run_batch_audio.
    """
    key = f"audio_{target_sr}"
    to_resample = [uid for uid in utt_ids if key not in utterances[uid]]
    if to_resample:
        for uid in tqdm(to_resample, desc=f"Resampling to {target_sr} Hz", unit="utt"):
            utt = utterances[uid]
            utterances[uid][key] = _resample(utt["audio"], utt["sr"], target_sr)
    return key


def _time_to_frame(t: float, fps: float, max_frame: int) -> int:
    return min(int(t * fps), max_frame)


def _slice_frames(hidden: np.ndarray, start: float, end: float,
                  fps: float) -> np.ndarray | None:
    """Slice encoder hidden states by word time boundary and mean-pool.

    hidden: (T, D) — encoder output for a full utterance
    Returns (D,) mean-pooled word embedding, or None if the slice is empty.
    """
    T = hidden.shape[0]
    f0 = _time_to_frame(start, fps, T)
    f1 = _time_to_frame(end,   fps, T)
    f1 = max(f0 + 1, f1)   # guarantee at least 1 frame
    f1 = min(f1, T)
    if f0 >= T:
        return None
    return hidden[f0:f1].mean(axis=0)


# ---------------------------------------------------------------------------
# Shared batch helper
# ---------------------------------------------------------------------------

def _run_batch_audio(
    model_name: str,
    utt_ids: list,
    start_utt: int,
    batch_size: int,
    word_records: list[dict],
    utterances: dict,
    utt_to_words: dict,
    completed_words: set,
    word_embeddings_list: list,
    encode_fn,          # callable(audio_arrays) → (B, T, D) np.ndarray
    fps: float,
    checkpoint_path,
    audio_key: str = "audio_resampled",
) -> int:
    """Shared batched loop for all audio encoder extractors.

    encode_fn takes a list of 1-D float32 numpy arrays (already resampled)
    and returns a (B, T, D) numpy array of encoder hidden states.

    audio_key is the key in utterances[uid] that holds the resampled array
    (set by _resample_utterances, e.g. "audio_16000" or "audio_24000").

    Returns the number of word-level errors encountered.
    """
    N_utt = len(utt_ids)
    errors = 0
    pbar = tqdm(range(start_utt, N_utt, batch_size), desc=model_name,
                unit="batch", total=(N_utt - start_utt + batch_size - 1) // batch_size)

    for batch_start in pbar:
        batch_ids = utt_ids[batch_start : batch_start + batch_size]
        try:
            audio_arrays = [utterances[uid][audio_key] for uid in batch_ids]
            hidden_batch = encode_fn(audio_arrays)          # (B, T, D)

            for b, utt_id in enumerate(batch_ids):
                hidden = hidden_batch[b]                    # (T, D)
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
# Whisper encoder word-level extraction
# ---------------------------------------------------------------------------

def extract_whisper_enc_word_embeddings(
    model_name: str,
    model_id: str,
    word_records: list[dict],
    utterances: dict,
    device: torch.device,
    fps: float = WHISPER_ENC_FPS,
    target_sr: int = WHISPER_SR,
    batch_size: int = 64,
    checkpoint_dir: Path = None,
) -> np.ndarray:
    """Extract word-level embeddings from a Whisper encoder (batched).

    Processes utterances in batches: runs the encoder once per batch to get
    (B, T, D) hidden states, then slices each utterance's word frames.
    """
    logger.info(f"Loading Whisper encoder: {model_id}")
    log_gpu_memory(f"before {model_name} load")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    log_gpu_memory(f"after {model_name} load")

    utt_to_words = _group_by_utt(word_records)
    utt_ids = list(utt_to_words.keys())
    audio_key = _resample_utterances(utterances, utt_ids, target_sr)

    checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint.pkl" if checkpoint_dir else None
    start_utt, word_embeddings_list = _load_checkpoint(checkpoint_path, model_name)
    completed_words = {idx for idx, _ in word_embeddings_list}

    def encode(audio_arrays):
        inputs = processor(audio_arrays, sampling_rate=target_sr,
                           return_tensors="pt")
        features = inputs["input_features"].to(device, dtype=torch.float16)
        with torch.no_grad():
            enc_out = model.encoder(features, output_hidden_states=False)
        return enc_out.last_hidden_state.float().cpu().numpy()  # (B, T, D)

    errors = _run_batch_audio(
        model_name, utt_ids, start_utt, batch_size,
        word_records, utterances, utt_to_words, completed_words,
        word_embeddings_list, encode, fps, checkpoint_path, audio_key,
    )

    del model
    release_vram(model_name)
    _remove_checkpoint(checkpoint_path)
    if errors:
        logger.warning(f"{model_name}: {errors} word-level errors")
    return _sort_embeddings(word_embeddings_list, len(word_records))


# ---------------------------------------------------------------------------
# Parakeet (FastConformer-CTC) word-level extraction
# ---------------------------------------------------------------------------

def extract_parakeet_word_embeddings(
    model_id: str,
    word_records: list[dict],
    utterances: dict,
    device: torch.device,
    fps: float = PARAKEET_FPS,
    target_sr: int = WHISPER_SR,
    batch_size: int = 32,
    checkpoint_dir: Path = None,
) -> np.ndarray:
    """Extract word-level embeddings from Parakeet FastConformer encoder (batched)."""
    model_name = "parakeet-ctc-0.6b"
    logger.info(f"Loading Parakeet: {model_id}")
    log_gpu_memory(f"before {model_name} load")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = ParakeetForCTC.from_pretrained(model_id, torch_dtype=torch.float32)
    model = model.to(device).eval()
    log_gpu_memory(f"after {model_name} load")

    utt_to_words = _group_by_utt(word_records)
    utt_ids = list(utt_to_words.keys())
    audio_key = _resample_utterances(utterances, utt_ids, target_sr)

    checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint.pkl" if checkpoint_dir else None
    start_utt, word_embeddings_list = _load_checkpoint(checkpoint_path, model_name)
    completed_words = {idx for idx, _ in word_embeddings_list}

    def encode(audio_arrays):
        inputs = feature_extractor(
            audio_arrays, sampling_rate=target_sr,
            return_tensors="pt", padding="longest",
        )
        input_key = next(k for k in inputs.keys() if "mask" not in k)
        input_tensor = inputs[input_key].to(device, dtype=torch.float32)
        with torch.no_grad():
            if hasattr(model, "parakeet") and hasattr(model.parakeet, "encoder"):
                enc_out = model.parakeet.encoder(input_tensor, output_hidden_states=True)
                last_hidden = enc_out.last_hidden_state
            else:
                out = model(input_tensor, output_hidden_states=True)
                last_hidden = out.hidden_states[-1]
        return last_hidden.float().cpu().numpy()  # (B, T, D)

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
# Mimi word-level extraction
# ---------------------------------------------------------------------------

def extract_mimi_word_embeddings(
    model_id: str,
    word_records: list[dict],
    utterances: dict,
    device: torch.device,
    fps: float = MIMI_FPS,
    target_sr: int = MIMI_SR,
    batch_size: int = 64,
    checkpoint_dir: Path = None,
) -> np.ndarray:
    """Extract word-level embeddings from Mimi encoder (batched, pre-RVQ hidden states)."""
    model_name = "mimi"
    logger.info(f"Loading Mimi: {model_id}")
    log_gpu_memory(f"before {model_name} load")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = MimiModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model = model.to(device).eval()
    log_gpu_memory(f"after {model_name} load")

    utt_to_words = _group_by_utt(word_records)
    utt_ids = list(utt_to_words.keys())
    audio_key = _resample_utterances(utterances, utt_ids, target_sr)

    checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint.pkl" if checkpoint_dir else None
    start_utt, word_embeddings_list = _load_checkpoint(checkpoint_path, model_name)
    completed_words = {idx for idx, _ in word_embeddings_list}

    def encode(audio_arrays):
        inputs = feature_extractor(
            raw_audio=audio_arrays, sampling_rate=target_sr,
            return_tensors="pt", padding=True,
        )
        input_values = inputs["input_values"].to(device)
        with torch.no_grad():
            enc_out = model.encoder(input_values)
        if isinstance(enc_out, torch.Tensor):
            # Channels-first (B, D, T) → transpose to (B, T, D)
            return enc_out.float().cpu().numpy().transpose(0, 2, 1)
        else:
            return enc_out.last_hidden_state.float().cpu().numpy()  # (B, T, D)

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
# Text LM word-level extraction
# ---------------------------------------------------------------------------

def extract_lm_word_embeddings(
    model_name: str,
    model_id: str,
    word_records: list[dict],
    device: torch.device,
    batch_size: int = 32,
    checkpoint_dir: Path = None,
) -> np.ndarray:
    """Extract word-level embeddings from a causal LM (batched).

    Processes utterances in batches: tokenizes B sentences together with
    padding, runs one forward pass, then uses per-sentence offset mappings
    to locate each word's tokens and mean-pools their hidden states.

    Context is preserved — the model sees the full sentence, not the word
    in isolation.
    """
    logger.info(f"Loading LM: {model_id}")
    log_gpu_memory(f"before {model_name} load")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True,
        )
    except Exception:
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True,
        )
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log_gpu_memory(f"after {model_name} load")

    utt_to_words = _group_by_utt(word_records)
    utt_ids = list(utt_to_words.keys())
    N_utt = len(utt_ids)

    checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint.pkl" if checkpoint_dir else None
    start_utt, word_embeddings_list = _load_checkpoint(checkpoint_path, model_name)
    completed_words = {idx for idx, _ in word_embeddings_list}

    errors = 0
    pbar = tqdm(range(start_utt, N_utt, batch_size), desc=model_name,
                unit="batch", total=(N_utt - start_utt + batch_size - 1) // batch_size)

    for batch_start in pbar:
        batch_ids = utt_ids[batch_start : batch_start + batch_size]
        sentences = [word_records[utt_to_words[uid][0]]["sentence"] for uid in batch_ids]

        try:
            enc = tokenizer(
                sentences,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TEXT_TOKENS,
                padding=True,
                return_offsets_mapping=True,
            )
            # offset_mapping: (B, T, 2) — padding tokens have (0, 0)
            offset_mapping = enc.pop("offset_mapping").tolist()
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
            hidden_batch = out.hidden_states[-1].float().cpu().numpy()  # (B, T, D)

            for b, utt_id in enumerate(batch_ids):
                hidden = hidden_batch[b]    # (T, D)
                om     = offset_mapping[b]  # [(ts, te), ...]
                for word_idx in utt_to_words[utt_id]:
                    if word_idx in completed_words:
                        continue
                    rec = word_records[word_idx]
                    char_s, char_e = rec["char_start"], rec["char_end"]
                    if char_s < 0:
                        errors += 1
                        continue
                    token_indices = [
                        t for t, (ts, te) in enumerate(om)
                        if ts < char_e and te > char_s and ts < te
                    ]
                    if not token_indices:
                        errors += 1
                        continue
                    emb = hidden[token_indices].mean(axis=0)
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

    del model
    release_vram(model_name)
    _remove_checkpoint(checkpoint_path)

    if errors:
        logger.warning(f"{model_name}: {errors} word-level errors")

    return _sort_embeddings(word_embeddings_list, len(word_records))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_by_utt(word_records: list[dict]) -> dict[str, list[int]]:
    """Map utterance ID → list of word record indices."""
    groups: dict[str, list[int]] = {}
    for idx, rec in enumerate(word_records):
        groups.setdefault(rec["utt_id"], []).append(idx)
    return groups


def _sort_embeddings(pairs: list[tuple[int, np.ndarray]],
                     n_words: int) -> np.ndarray:
    """Convert list of (word_idx, embedding) pairs to ordered (N, D) array.

    Words with no embedding (failed slicing) get a zero vector, which will
    produce NaN after L2 normalisation and be excluded from CKA.
    """
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    if not pairs_sorted:
        raise RuntimeError("No word embeddings were collected.")
    D = pairs_sorted[0][1].shape[0]
    result = np.zeros((n_words, D), dtype=np.float32)
    for idx, emb in pairs_sorted:
        result[idx] = emb
    return result


# ---------------------------------------------------------------------------
# Minibatch CKA  (identical to representation_analysis.py)
# ---------------------------------------------------------------------------

def _hsic1_batch(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased HSIC estimator via Szekely & Rizzo (2014) kernel centering,
    as described in Kornblith et al. (2019) and Murphy et al. (2024)."""
    n = X.shape[0]
    assert n >= 4, f"Minibatch size must be >= 4 for HSIC_1; got {n}"
    K = X @ X.T
    L = Y @ Y.T
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    KL = K @ L
    ones = np.ones(n)
    # Expanding (K - mean_K)(L - mean_L) = K·L - K·mean_L - mean_K·L + mean_K·mean_L
    # gives three correction terms (subtracting both means double-subtracts the
    # grand mean, so we add it back as term2):
    term1 = np.trace(KL)
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * (ones @ KL @ ones)
    return float((term1 + term2 - term3) / (n * (n - 3)))


def minibatch_cka(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = MINIBATCH_SIZE,
    seed: int = MINIBATCH_SEED,
) -> tuple[float, float]:
    """Minibatch linear CKA. Returns (cka_score, hsic_xy_std)."""
    # Filter out zero rows (words with no embedding)
    valid = (np.linalg.norm(X, axis=1) > 1e-10) & (np.linalg.norm(Y, axis=1) > 1e-10)
    X, Y = X[valid], Y[valid]

    N = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)
    hsic_xy, hsic_xx, hsic_yy = [], [], []

    for start in range(0, N - batch_size + 1, batch_size):
        idx = indices[start : start + batch_size]
        Xb = X[idx].astype(np.float64)
        Yb = Y[idx].astype(np.float64)
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-10
        Yb /= np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-10
        hsic_xy.append(_hsic1_batch(Xb, Yb))
        hsic_xx.append(_hsic1_batch(Xb, Xb))
        hsic_yy.append(_hsic1_batch(Yb, Yb))

    if not hsic_xy:
        return 0.0, 0.0

    mean_xy = float(np.mean(hsic_xy))
    std_xy  = float(np.std(hsic_xy))
    denom = np.sqrt(max(float(np.mean(hsic_xx)), 0.0) * max(float(np.mean(hsic_yy)), 0.0))
    score = float(mean_xy / denom) if denom > 1e-10 else 0.0
    return score, std_xy


# ---------------------------------------------------------------------------
# PCA / eigenvalue analysis (identical to representation_analysis.py)
# ---------------------------------------------------------------------------

def pca_eigenvalues(X: np.ndarray, n_components: int = PCA_COMPONENTS) -> np.ndarray:
    from sklearn.utils.extmath import randomized_svd
    valid = np.linalg.norm(X, axis=1) > 1e-10
    Xv = X[valid].astype(np.float64)
    Xc = Xv - Xv.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    if not np.isfinite(Xc).all():
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    _, s, _ = randomized_svd(
        Xc / np.sqrt(n - 1),
        n_components=min(n_components, min(Xc.shape) - 1),
        n_oversamples=10, n_iter=4, random_state=MINIBATCH_SEED,
    )
    eigs = s ** 2
    eigs /= eigs.sum()
    return eigs


def effective_rank(eigenvalues: np.ndarray) -> float:
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_cka_heatmap(cka_matrix: np.ndarray, names: list, plots_dir: Path,
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
            text_color = "black" if val < 0.7 else "white"
            if cka_results is not None and i != j:
                key = f"{names[min(i,j)]} vs {names[max(i,j)]}"
                std = cka_results.get(key, {}).get("hsic_std", None)
                label = f"{val:.3f}\n±{std:.3f}" if std is not None else f"{val:.3f}"
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

    ax.set_title("Word-Level Pairwise Minibatch Linear CKA\n(LJSpeech + TextGrid alignments)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    path = plots_dir / "word_cka_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_eigenspectra_overlay(eigenvalues: dict, plots_dir: Path,
                               max_pc: int = PCA_PLOT_MAX_PC):
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, eigs_full in eigenvalues.items():
        eigs = eigs_full[:max_pc]
        color = MODEL_COLORS.get(name, "#555555")
        k = np.arange(1, len(eigs) + 1)
        ax.plot(k, eigs, color=color, linewidth=2, label=name, alpha=0.85)
    ax.set_xlabel(f"Principal Component (1–{max_pc})", fontsize=11)
    ax.set_ylabel("Fraction of Variance", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlim(1, max_pc)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"Word-Level PCA Eigenvalue Spectra  ·  first {max_pc} PCs",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = plots_dir / "word_eigenspectra_overlay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_effective_rank_bar(eigenvalues: dict, plots_dir: Path):
    _apply_style()
    names = list(eigenvalues.keys())
    ranks = [effective_rank(eigenvalues[n]) for n in names]
    colors = [MODEL_COLORS.get(n, "#555555") for n in names]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    bars = ax.bar(range(len(names)), ranks, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("Word-Level Effective Rank per Model", fontsize=12, fontweight="bold")
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{rank:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = plots_dir / "word_effective_rank.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------

def save_summary_tables(cka_matrix: np.ndarray, cka_results: dict,
                        names: list, eigenvalues: dict,
                        embeddings: dict, data_dir: Path):
    path1 = data_dir / "word_cka_matrix.csv"
    with open(path1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, row_name in enumerate(names):
            w.writerow([row_name] + [f"{cka_matrix[i,j]:.6f}" for j in range(len(names))])
    logger.info(f"  Saved → {path1}")

    path2 = data_dir / "word_cka_minibatch_std.csv"
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "cka_score", "hsic_std"])
        for pair, vals in cka_results.items():
            w.writerow([pair, f"{vals['score']:.6f}", f"{vals['hsic_std']:.6f}"])
    logger.info(f"  Saved → {path2}")

    path3 = data_dir / "word_pca_summary.csv"
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
    p = argparse.ArgumentParser(description="Word-level cross-modal CKA on LJSpeech + TextGrids")
    p.add_argument("--textgrid_dir", default=None, type=Path,
                   help="Directory containing .TextGrid files (one per LJSpeech utterance). "
                        "If omitted, defaults to <root_dir>/WordData/textgrids and the script "
                        "will attempt to download from Kaggle automatically.")
    p.add_argument("--ljspeech_dir", default=None, type=Path,
                   help="Path to the LJSpeech-1.1 root directory (must contain metadata.csv "
                        "and wavs/). If omitted, defaults to <root_dir>/WordData/LJSpeech-1.1 "
                        "and downloads automatically if not present.")
    p.add_argument("--root_dir", default=".", type=Path,
                   help="Root directory for WordData/, WordPlots/, logs/")
    p.add_argument("--max_words", default=None, type=int,
                   help="Cap total word records (useful for testing)")
    p.add_argument("--batch_size", default=2048, type=int,
                   help="Minibatch size for CKA computation")
    p.add_argument("--whisper_batch_size", default=64, type=int,
                   help="Number of utterances per batch for Whisper encoder extraction")
    p.add_argument("--parakeet_batch_size", default=32, type=int,
                   help="Number of utterances per batch for Parakeet extraction")
    p.add_argument("--mimi_batch_size", default=64, type=int,
                   help="Number of utterances per batch for Mimi extraction")
    p.add_argument("--lm_batch_size", default=32, type=int,
                   help="Number of utterances per batch for LM extraction")
    p.add_argument("--skip_extraction", action="store_true",
                   help="Skip extraction even if no cache exists")
    p.add_argument("--pca_components", default=50, type=int)
    p.add_argument("--pca_plot_max_pc", default=10, type=int)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root = Path(args.root_dir)
    data_dir  = root / "WordData"
    plots_dir = root / "WordPlots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(root / "logs")
    logger.info("=" * 60)
    logger.info("Word-Level Representation Analysis — run started")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("=" * 60)

    device = get_device()

    # ------------------------------------------------------------------
    # 0. Ensure TextGrids are present (download from Kaggle if needed)
    # ------------------------------------------------------------------
    ljspeech_dir = args.ljspeech_dir or (data_dir / "LJSpeech-1.1")
    try:
        ljspeech_dir = ensure_ljspeech(ljspeech_dir)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    textgrid_dir = args.textgrid_dir or (data_dir / "textgrids")
    try:
        textgrid_dir = ensure_textgrids(textgrid_dir)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load LJSpeech + build word records
    # ------------------------------------------------------------------
    # Include max_words in the cache filename so that runs with different
    # --max_words arguments don't silently reuse each other's cached records.
    cache_suffix = f"_max{args.max_words}" if args.max_words else ""
    word_records_path = data_dir / f"word_records{cache_suffix}.json"

    if word_records_path.exists():
        logger.info(f"Loading cached word records from {word_records_path}")
        with open(word_records_path) as f:
            word_records = json.load(f)
        logger.info(f"Loaded {len(word_records):,} word records")
        # Still need utterances for audio extraction
        utterances = load_ljspeech(ljspeech_dir)
    else:
        with timer("Load LJSpeech"):
            utterances = load_ljspeech(ljspeech_dir)
        with timer("Build word records"):
            word_records = build_word_records(
                utterances, textgrid_dir,
                max_words=args.max_words,
            )
        with open(word_records_path, "w") as f:
            json.dump(word_records, f)
        logger.info(f"Cached word records → {word_records_path}")

    N = len(word_records)
    logger.info(f"Total word tokens: {N:,}")

    # ------------------------------------------------------------------
    # 2. Extract / load embeddings
    # ------------------------------------------------------------------
    embeddings = {}
    for model_name, cfg in MODELS.items():
        cache_path = data_dir / f"word_embeddings_{model_name}.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached word embeddings: {model_name}")
            with open(cache_path, "rb") as f:
                embeddings[model_name] = pickle.load(f)
            logger.info(f"  Shape: {embeddings[model_name].shape}")
            continue

        if args.skip_extraction:
            logger.warning(f"--skip_extraction but no cache for {model_name}, skipping")
            continue

        logger.info("=" * 60)
        logger.info(f"Extracting word embeddings: {model_name}  "
                    f"[{cfg['params']} | {cfg['arch']}]")
        logger.info("=" * 60)

        with timer(f"Extract {model_name}"):
            try:
                modality = cfg["modality"]
                if modality == "audio-whisper-enc":
                    emb = extract_whisper_enc_word_embeddings(
                        model_name, cfg["hf_id"], word_records, utterances,
                        device, fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.whisper_batch_size,
                        checkpoint_dir=data_dir,
                    )
                elif modality == "audio-parakeet":
                    emb = extract_parakeet_word_embeddings(
                        cfg["hf_id"], word_records, utterances,
                        device, fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.parakeet_batch_size,
                        checkpoint_dir=data_dir,
                    )
                elif modality == "audio-mimi":
                    emb = extract_mimi_word_embeddings(
                        cfg["hf_id"], word_records, utterances,
                        device, fps=cfg["fps"], target_sr=cfg["target_sr"],
                        batch_size=args.mimi_batch_size,
                        checkpoint_dir=data_dir,
                    )
                else:  # text
                    emb = extract_lm_word_embeddings(
                        model_name, cfg["hf_id"], word_records,
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

    # ------------------------------------------------------------------
    # 3. PCA eigenvalue analysis
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PCA eigenvalue analysis")
    logger.info("=" * 60)
    eigenvalues = {}
    with timer("PCA"):
        for name, X in embeddings.items():
            eigs = pca_eigenvalues(X, n_components=args.pca_components)
            eigenvalues[name] = eigs
            er = effective_rank(eigs)
            logger.info(f"  {name:<22}  dim={X.shape[1]:>5}  eff_rank={er:.1f}")

    with timer("PCA plots"):
        plot_eigenspectra_overlay(eigenvalues, plots_dir, max_pc=args.pca_plot_max_pc)
        plot_effective_rank_bar(eigenvalues, plots_dir)

    # ------------------------------------------------------------------
    # 4. Pairwise minibatch CKA
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Pairwise word-level minibatch CKA  (batch_size={args.batch_size}, N={N:,})")
    logger.info("=" * 60)

    n = len(names)
    n_pairs = n * (n - 1) // 2
    cka_matrix = np.zeros((n, n))
    cka_results = {}

    with timer("Pairwise CKA"):
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
                score, hsic_std = minibatch_cka(embeddings[names[i]], embeddings[names[j]],
                                                batch_size=args.batch_size)
                cka_matrix[i, j] = score
                key = f"{names[i]} vs {names[j]}"
                cka_results[key] = {"score": score, "hsic_std": hsic_std}
                logger.info(f"  CKA({names[i]}, {names[j]}) = {score:.4f}  "
                            f"(hsic_std={hsic_std:.4f}, {time.perf_counter()-t0:.1f}s)")
                pair_bar.update(1)
        pair_bar.close()

    with timer("CKA plots"):
        plot_cka_heatmap(cka_matrix, names, plots_dir, cka_results=cka_results)

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    with timer("Save tables"):
        save_summary_tables(cka_matrix, cka_results, names, eigenvalues, embeddings, data_dir)

    results_json = {
        "n_words": N,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "fps" and kk != "target_sr"}
                   for k, v in MODELS.items()},
        "embedding_shapes": {k: list(v.shape) for k, v in embeddings.items()},
        "cka_scores": cka_results,
        "cka_matrix": cka_matrix.tolist(),
        "eigenvalue_spectra": {k: v.tolist() for k, v in eigenvalues.items()},
        "effective_ranks": {k: effective_rank(v) for k, v in eigenvalues.items()},
    }
    json_path = data_dir / "word_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results JSON → {json_path}")

    logger.info("Run complete.")


if __name__ == "__main__":
    main()
