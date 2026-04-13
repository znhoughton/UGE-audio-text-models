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
  logs/    — timestamped run logs (.log)

Usage:
    # Full run (all LibriSpeech splits, ~280k utterances)
    python representation_analysis.py

    # Quick test run
    python representation_analysis.py --splits test --max_samples 500

    # Re-run analysis using cached embeddings (skip re-extraction)
    python representation_analysis.py --skip_extraction

Models:
    - openai/whisper-base                       (audio, 74M)
    - nvidia/parakeet-ctc-0.6b                  (audio, 600M, FastConformer-CTC)
    - znhoughton/opt-babylm-125m-20eps-seed964  (text, 125M, OPT, BabyLM)
    - znhoughton/opt-babylm-1.3b-20eps-seed964  (text, 1.3B, OPT, BabyLM)
    - allenai/OLMo-2-1124-7B                    (text, 7B, OLMo-2, Dolma)
    - EleutherAI/pythia-6.9b                    (text, 6.9B, GPT-NeoX, The Pile)
"""

import argparse
import csv
import json
import logging
import os
import pickle
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# HuggingFace memory / shard loading settings
# Must be set before any datasets import to take effect.
# ---------------------------------------------------------------------------

# Cap how much RAM datasets will try to use for in-memory operations.
# Each shard is ~450-500MB on disk, ~1-1.5GB in RAM after Arrow expansion.
# 60GB gives comfortable headroom for parallel shard loading without
# trying to hold all 63 shards (~79GB) simultaneously.
os.environ.setdefault("HF_DATASETS_MAX_IN_MEMORY_SIZE", str(60 * 1024 ** 3))  # 60 GB
os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", str(60 * 1024 ** 3))
os.environ.setdefault("DATASETS_VERBOSITY", "warning")

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
    AutoFeatureExtractor,
    ParakeetForCTC,
    WhisperModel,
    WhisperProcessor,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("repr_analysis")


def setup_logging(log_dir: Path) -> Path:
    """
    Configure logging to write simultaneously to stdout and a timestamped log file.
    Returns the path of the log file created.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — full DEBUG level, captures everything
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above (keeps stdout readable)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return log_path


@contextmanager
def timer(label: str):
    """Context manager that logs wall-clock time for a labelled block."""
    logger.info(f"[START] {label}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            fmt = f"{int(hrs)}h {int(mins)}m {secs:.1f}s"
        elif mins > 0:
            fmt = f"{int(mins)}m {secs:.1f}s"
        else:
            fmt = f"{secs:.2f}s"
        logger.info(f"[DONE]  {label}  ({fmt})")


def log_gpu_memory(label: str = ""):
    """Log current and peak GPU VRAM usage if CUDA is available."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    peak      = torch.cuda.max_memory_allocated() / 1024 ** 3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    tag = f"[{label}] " if label else ""
    logger.debug(
        f"{tag}GPU memory — "
        f"allocated: {allocated:.2f} GB  "
        f"reserved: {reserved:.2f} GB  "
        f"peak: {peak:.2f} GB  "
        f"total: {total:.2f} GB"
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
    "parakeet-ctc-0.6b": {
        "hf_id": "nvidia/parakeet-ctc-0.6b",
        "modality": "audio-parakeet",
        "params": "600M",
        "arch": "FastConformer-CTC",
        "corpus": "Granary (64k hrs English)",
    },
}

ALL_SPLITS = [
    "train.100",       # clean 100hr train (HF: config=clean, split=train.100)
    "train.360",       # clean 360hr train (HF: config=clean, split=train.360)
    "train.500",       # other 500hr train (HF: config=other, split=train.500)
    "validation",      # clean validation  (HF: config=clean, split=validation)
    "validation.other",# other validation  (HF: config=other, split=validation)
    "test",            # clean test        (HF: config=clean, split=test)
    "test.other",      # other test        (HF: config=other, split=test)
]

# Maps our logical split names to (hf_config, hf_split_name)
SPLIT_CONFIG_MAP = {
    "train.100":        ("clean", "train.100"),
    "train.360":        ("clean", "train.360"),
    "train.500":        ("other", "train.500"),
    "validation":       ("clean", "validation"),
    "validation.other": ("other", "validation"),
    "test":             ("clean", "test"),
    "test.other":       ("other", "test"),
}

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30
MAX_TEXT_TOKENS = 512
PCA_COMPONENTS = 50

MINIBATCH_SIZE = 2048
MINIBATCH_SEED = 42

N_STABILITY_SUBSETS = 10
STABILITY_SUBSET_FRAC = 0.8

MODEL_COLORS = {
    "whisper-base":       "#1565C0",
    "babylm-125m":        "#E65100",
    "babylm-1.3b":        "#F9A825",
    "olmo-7b":            "#2E7D32",
    "pythia-6.9b":        "#6A1B9A",
    "parakeet-ctc-0.6b":  "#00838F",   # teal — second audio model
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024 ** 3
        logger.info(f"GPU: {props.name}  ({total_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — running on CPU (will be very slow for 7B models)")
    return device


def make_dirs(root: Path):
    (root / "Data").mkdir(parents=True, exist_ok=True)
    (root / "Plots").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)


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
    logger.info(f"{'='*60}")
    logger.info(f"Loading LibriSpeech splits: {splits}")
    logger.info(f"{'='*60}")

    parts = []
    for split in splits:
        if split not in SPLIT_CONFIG_MAP:
            raise ValueError(
                f"Unknown split '{split}'. Valid options: {list(SPLIT_CONFIG_MAP.keys())}"
            )
        hf_config, hf_split = SPLIT_CONFIG_MAP[split]
        logger.info(f"  Fetching {split} (config={hf_config}, split={hf_split})…")
        try:
            ds = load_dataset(
                "openslr/librispeech_asr",
                hf_config,
                split=hf_split,
                streaming=False,
                trust_remote_code=True,
                # 8 parallel workers gives meaningful I/O parallelism on a
                # network filesystem without the memory spike of loading all
                # 63 shards simultaneously. At ~1.25GB per shard in RAM,
                # 8 workers peaks at ~10GB — well within our 60GB budget.
                num_proc=8,
            )
        except ValueError as e:
            logger.error(
                f"Failed to load split '{split}' (config={hf_config}, split={hf_split}): {e}\n"
                f"The HuggingFace split name may differ from what we expect. "
                f"Try loading the dataset manually to check available splits:\n"
                f"  from datasets import get_dataset_split_names\n"
                f"  print(get_dataset_split_names('openslr/librispeech_asr', '{hf_config}'))"
            )
            raise
        logger.info(f"  {split}: {len(ds):,} samples")
        parts.append(ds)

    logger.info(f"Concatenating {len(parts)} splits…")
    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    logger.info(f"Total before sampling: {len(dataset):,} samples")

    if max_samples and max_samples < len(dataset):
        rng = random.Random(MINIBATCH_SEED)
        indices = sorted(rng.sample(range(len(dataset)), max_samples))
        dataset = dataset.select(indices)
        logger.info(f"Subsampled to {len(dataset):,} samples (seed={MINIBATCH_SEED})")

    # Extract texts up front so we hold only the text in RAM.
    # The audio waveforms remain on disk and are streamed per-sample
    # during embedding extraction — we never load all audio at once.
    logger.info("Extracting transcripts…")
    texts = [s["text"].strip() for s in tqdm(dataset, desc="reading transcripts", unit="utt")]
    logger.info(f"Final dataset size: {len(texts):,} utterances")
    logger.debug(f"Example transcript: '{texts[0][:80]}…'")
    return dataset, texts


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_whisper_embeddings(model_id: str, dataset, device: torch.device) -> np.ndarray:
    """Mean-pool Whisper encoder hidden states over time frames."""
    logger.info(f"Loading Whisper model: {model_id}")
    log_gpu_memory("before Whisper load")

    try:
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to(device).eval()
    except Exception as e:
        logger.error(f"Failed to load Whisper model '{model_id}': {e}")
        raise

    log_gpu_memory("after Whisper load")
    logger.info(f"Extracting Whisper embeddings for {len(dataset):,} samples…")

    embeddings = []
    errors = 0
    for sample in tqdm(dataset, desc="whisper-base", unit="utt"):
        try:
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
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping sample due to error: {e}")

    if errors:
        logger.warning(f"Whisper: skipped {errors} samples due to errors")

    del model
    torch.cuda.empty_cache()
    log_gpu_memory("after Whisper unload")

    result = np.stack(embeddings)
    logger.info(f"Whisper embeddings shape: {result.shape}")
    return result


def extract_parakeet_embeddings(model_id: str, dataset, device: torch.device) -> np.ndarray:
    """
    Extract encoder embeddings from Parakeet CTC via the HuggingFace transformers
    Parakeet integration (requires transformers >= 4.48).

    Parakeet's FastConformer encoder applies 8x depthwise-separable convolutional
    downsampling before its conformer blocks, so the output sequence is much shorter
    than Whisper's. We mean-pool across the remaining time frames, exactly as we do
    for Whisper, to get one fixed-size vector per utterance.
    """
    logger.info(f"Loading Parakeet model: {model_id}")
    log_gpu_memory("before Parakeet load")

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = ParakeetForCTC.from_pretrained(model_id, torch_dtype=torch.float16)
        model = model.to(device).eval()
    except Exception as e:
        logger.error(
            f"Failed to load Parakeet model '{model_id}': {e}\n"
            f"Make sure transformers >= 4.48 is installed: pip install --upgrade transformers"
        )
        raise

    log_gpu_memory("after Parakeet load")
    logger.info(f"Extracting Parakeet embeddings for {len(dataset):,} samples…")

    embeddings = []
    errors = 0
    for sample in tqdm(dataset, desc="parakeet-ctc-0.6b", unit="utt"):
        try:
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            if sr != SAMPLE_RATE:
                audio = audio[:: int(sr / SAMPLE_RATE)]
            audio = audio[: MAX_AUDIO_SECONDS * SAMPLE_RATE]

            inputs = feature_extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(device, dtype=torch.float16)

            with torch.no_grad():
                out = model(input_values, output_hidden_states=True)
                emb = out.hidden_states[-1].mean(dim=1).squeeze(0).float().cpu()
            embeddings.append(emb.numpy())
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping sample due to error: {e}")

    if errors:
        logger.warning(f"Parakeet: skipped {errors} samples due to errors")

    del model
    torch.cuda.empty_cache()
    log_gpu_memory("after Parakeet unload")

    result = np.stack(embeddings)
    logger.info(f"Parakeet embeddings shape: {result.shape}")
    return result


def extract_lm_embeddings(
    model_id: str,
    texts: list,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Mean-pool last hidden layer over non-padding tokens."""
    logger.info(f"Loading LM: {model_id}")
    log_gpu_memory(f"before {model_id.split('/')[-1]} load")

    load_kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception:
        try:
            model = AutoModel.from_pretrained(model_id, **load_kwargs)
        except Exception as e:
            logger.error(
                f"Failed to load model '{model_id}': {e}\n"
                f"If this is an OOM error, try reducing --lm_batch_size (currently {batch_size})"
            )
            raise

    model = model.to(device).eval()
    log_gpu_memory(f"after {model_id.split('/')[-1]} load")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer for '{model_id}': {e}")
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug(f"Set pad_token = eos_token ({tokenizer.eos_token!r})")

    n_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(
        f"Extracting embeddings: {len(texts):,} samples, "
        f"batch_size={batch_size}, n_batches={n_batches:,}"
    )

    embeddings = []
    errors = 0
    label = model_id.split("/")[-1]

    for i in tqdm(range(0, len(texts), batch_size), desc=label, unit="batch", total=n_batches):
        batch = texts[i : i + batch_size]
        try:
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
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"CUDA OOM at batch {i//batch_size}/{n_batches}. "
                    f"Try reducing --lm_batch_size (currently {batch_size}). "
                    f"Embeddings extracted so far: {len(embeddings) * batch_size:,}"
                )
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch {i//batch_size} due to error: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch {i//batch_size} due to error: {e}")

    if errors:
        logger.warning(f"{label}: skipped {errors} batches due to errors")

    del model
    torch.cuda.empty_cache()
    log_gpu_memory(f"after {label} unload")

    result = np.concatenate(embeddings, axis=0)
    logger.info(f"{label} embeddings shape: {result.shape}")
    return result


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
    logger.info(f"  Saved → {path}")


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
    logger.info(f"  Saved → {path}")


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
    logger.info(f"  Saved → {path}")


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
    logger.info(f"  Saved → {path}")


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

    audio_indices = [
        i for i, name in enumerate(names)
        if MODELS[name]["modality"] in ("audio", "audio-parakeet")
    ]
    if audio_indices:
        boundary = max(audio_indices) + 0.5
        ax.axhline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.axvline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_title(
        "Pairwise Minibatch Linear CKA\n"
        "Dashed line separates audio models (Whisper, Parakeet) from text models",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    path = plots_dir / "cka_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_cka_bar_cross_modal(cka_matrix: np.ndarray, names: list, plots_dir: Path):
    """
    Grouped bar chart: for each LLM, show its CKA with each audio model side by side.
    This makes it easy to see whether both audio models agree on which LLMs are most similar.
    """
    _apply_style()
    audio_names = [
        n for n in names
        if MODELS[n]["modality"] in ("audio", "audio-parakeet")
    ]
    lm_names = [
        n for n in names
        if MODELS[n]["modality"] not in ("audio", "audio-parakeet")
    ]
    if not audio_names or not lm_names:
        return

    x = np.arange(len(lm_names))
    n_audio = len(audio_names)
    width = 0.7 / n_audio   # total bar group width = 0.7, split evenly

    fig, ax = plt.subplots(figsize=(max(7, len(lm_names) * 1.8), 5))

    for k, audio_name in enumerate(audio_names):
        a_idx = names.index(audio_name)
        scores = [cka_matrix[a_idx, names.index(lm)] for lm in lm_names]
        offsets = x + (k - (n_audio - 1) / 2) * width
        color = MODEL_COLORS.get(audio_name, "#555")
        bars = ax.bar(
            offsets, scores,
            width=width * 0.9,
            color=color,
            label=audio_name,
            edgecolor="white", linewidth=0.8,
        )
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{score:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(lm_names, fontsize=10)
    ax.set_ylim(0, min(1.0, max(
        cka_matrix[names.index(a), names.index(lm)]
        for a in audio_names for lm in lm_names
    ) * 1.35 + 0.05))
    ax.set_ylabel("CKA with audio model", fontsize=11)
    ax.set_title(
        "Cross-Modal CKA: Audio Models vs. Each LLM\n"
        "Higher = more similar representational geometry to that audio encoder",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.9, title="Audio model")
    plt.tight_layout()
    path = plots_dir / "cka_cross_modal_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


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
    logger.info(f"  Saved → {path}")


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
    logger.info(f"  Saved → {path1}")

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
        logger.info(f"  Saved → {path2}")

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
    logger.info(f"  Saved → {path3}")

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
        logger.info(f"  Saved → {path4}")


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
        "--splits", nargs="+", default=ALL_SPLITS, choices=list(SPLIT_CONFIG_MAP.keys()),
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

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_path = setup_logging(root / "logs")
    logger.info("=" * 60)
    logger.info("Representation Analysis — run started")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Root dir: {root.resolve()}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("=" * 60)

    run_start = time.perf_counter()
    device = get_device()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    with timer("Load LibriSpeech"):
        dataset, texts = load_librispeech(args.splits, args.max_samples)
    N = len(texts)

    # ------------------------------------------------------------------
    # 2. Extract / load embeddings (one model at a time)
    # ------------------------------------------------------------------
    embeddings = {}
    for model_name, cfg in MODELS.items():
        cache_path = data_dir / f"embeddings_{model_name}.pkl"

        if args.skip_extraction and cache_path.exists():
            logger.info(f"Loading cached embeddings: {model_name}")
            with open(cache_path, "rb") as f:
                embeddings[model_name] = pickle.load(f)
            logger.info(f"  Shape: {embeddings[model_name].shape}")
            continue

        logger.info("=" * 60)
        logger.info(f"Extracting: {model_name}  [{cfg['params']} | {cfg['arch']} | {cfg['corpus']}]")
        logger.info("=" * 60)

        with timer(f"Extract {model_name}"):
            try:
                if cfg["modality"] == "audio":
                    emb = extract_whisper_embeddings(cfg["hf_id"], dataset, device)
                elif cfg["modality"] == "audio-parakeet":
                    emb = extract_parakeet_embeddings(cfg["hf_id"], dataset, device)
                else:
                    emb = extract_lm_embeddings(
                        cfg["hf_id"], texts, device, batch_size=args.lm_batch_size
                    )
            except Exception as e:
                logger.error(
                    f"Extraction failed for {model_name}: {e}\n"
                    f"Skipping this model and continuing with others."
                )
                continue

        embeddings[model_name] = emb
        with open(cache_path, "wb") as f:
            pickle.dump(emb, f)
        logger.info(f"Cached embeddings → {cache_path}  ({cache_path.stat().st_size / 1024**2:.1f} MB)")

    names = list(embeddings.keys())
    logger.info(f"Models with embeddings: {names}")

    # ------------------------------------------------------------------
    # 3. PCA eigenvalue analysis
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PCA eigenvalue analysis")
    logger.info("=" * 60)

    eigenvalues = {}
    with timer("PCA eigenvalue computation"):
        for name, X in embeddings.items():
            eigenvalues[name] = pca_eigenvalues(X, n_components=args.pca_components)
            er = effective_rank(eigenvalues[name])
            logger.info(
                f"  {name:<22}  dim={X.shape[1]:>5}  "
                f"n_samples={X.shape[0]:>7,}  eff_rank={er:.1f}"
            )

    with timer("PCA plots"):
        plot_eigenspectra(eigenvalues, plots_dir)
        plot_eigenspectra_overlay(eigenvalues, plots_dir)
        plot_effective_rank_bar(eigenvalues, plots_dir)
        try:
            from sklearn.decomposition import PCA  # noqa: F401
            plot_pca_scatter(embeddings, plots_dir)
        except ImportError:
            logger.warning("scikit-learn not installed — skipping PCA scatter (pip install scikit-learn)")

    # ------------------------------------------------------------------
    # 4. Pairwise minibatch CKA
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Pairwise minibatch CKA  (batch_size={args.batch_size}, N={N:,})")
    logger.info("=" * 60)

    n = len(names)
    n_pairs = n * (n - 1) // 2
    cka_matrix = np.zeros((n, n))
    cka_results = {}

    with timer("Pairwise CKA computation"):
        pair_bar = tqdm(total=n_pairs, desc="CKA pairs", unit="pair")
        for i in range(n):
            for j in range(n):
                if i == j:
                    cka_matrix[i, j] = 1.0
                    continue
                if j < i:
                    cka_matrix[i, j] = cka_matrix[j, i]
                    continue
                t_pair = time.perf_counter()
                score = minibatch_cka(
                    embeddings[names[i]], embeddings[names[j]],
                    batch_size=args.batch_size,
                )
                elapsed_pair = time.perf_counter() - t_pair
                cka_matrix[i, j] = score
                key = f"{names[i]} vs {names[j]}"
                cka_results[key] = score
                logger.info(
                    f"  CKA({names[i]}, {names[j]}) = {score:.4f}"
                    f"  ({elapsed_pair:.1f}s)"
                )
                pair_bar.update(1)
        pair_bar.close()

    with timer("CKA plots"):
        plot_cka_heatmap(cka_matrix, names, plots_dir)
        plot_cka_bar_cross_modal(cka_matrix, names, plots_dir)

    # ------------------------------------------------------------------
    # 5. Outlier stability check  (Horoi et al.)
    # ------------------------------------------------------------------
    stability_results = {}
    if not args.skip_stability:
        logger.info("=" * 60)
        logger.info(
            f"Outlier stability check  "
            f"({N_STABILITY_SUBSETS} subsets × {int(STABILITY_SUBSET_FRAC*100)}% of data)"
        )
        logger.info("=" * 60)

        upper_pairs = [
            (names[i], names[j])
            for i in range(n) for j in range(n) if j > i
        ]
        n_stability_jobs = len(upper_pairs) * N_STABILITY_SUBSETS

        with timer("Stability check"):
            stab_bar = tqdm(
                total=n_stability_jobs,
                desc="Stability subsets",
                unit="subset",
            )
            for name_a, name_b in upper_pairs:
                pair_key = f"{name_a} vs {name_b}"
                stats = cka_stability_check(
                    embeddings[name_a], embeddings[name_b],
                    batch_size=args.batch_size,
                )
                stability_results[pair_key] = stats
                flag = "  ⚠ HIGH VARIANCE" if stats["std"] > 0.03 else ""
                logger.info(
                    f"  {pair_key:<42}  "
                    f"mean={stats['mean']:.4f}  std={stats['std']:.4f}{flag}"
                )
                stab_bar.update(N_STABILITY_SUBSETS)
            stab_bar.close()

        with timer("Stability plot"):
            plot_stability(stability_results, plots_dir)
    else:
        logger.info("Stability check skipped (--skip_stability)")

    # ------------------------------------------------------------------
    # 6. Save all outputs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Saving summary tables and results JSON")
    logger.info("=" * 60)

    with timer("Save tables"):
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
    logger.info(f"Results JSON → {json_path}")

    # ------------------------------------------------------------------
    # 7. Console summary
    # ------------------------------------------------------------------
    print_summary(cka_matrix, names, stability_results)

    total_elapsed = time.perf_counter() - run_start
    total_mins, total_secs = divmod(total_elapsed, 60)
    total_hrs, total_mins = divmod(total_mins, 60)
    logger.info("=" * 60)
    logger.info(
        f"Run complete — total time: "
        f"{int(total_hrs)}h {int(total_mins)}m {total_secs:.1f}s"
    )
    logger.info(f"Data/   → {data_dir.resolve()}")
    logger.info(f"Plots/  → {plots_dir.resolve()}")
    logger.info(f"Log     → {log_path}")


if __name__ == "__main__":
    main()
