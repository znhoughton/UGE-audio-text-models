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
    - openai/whisper-base          (audio encoder, 74M)
    - openai/whisper-base          (audio decoder, 74M — same checkpoint, decoder side)
    - nvidia/parakeet-ctc-0.6b     (audio encoder, 600M, FastConformer-CTC)
                                   NOTE: Parakeet CTC has no decoder; decoder extraction skipped.
    - kyutai/mimi                  (audio codec encoder, ~85M, conv+transformer, 12.5Hz)
                                   NOTE: pure audio codec, no text signal. Extracts pre-RVQ
                                   encoder hidden state — the most purely acoustic model here.
    - Qwen/Qwen3-TTS-12Hz-1.7B-Base  (TTS, 1.7B, discrete multi-codebook LM)
                                     NOTE: uses qwen-tts package, not transformers.
                                     Extracts LM backbone hidden states on input text.
    - mistralai/Voxtral-Mini-3B-2507  (speech understanding model, 3B)
    - bosonai/higgs-audio-v2-generation-3B-base  (TTS, ~5.8B, Llama-3.2-3B + DualFFN)
                                     NOTE: text-to-audio generation model (not audio-in).
                                     Extracts LLM backbone hidden states on text input.
    - znhoughton/opt-babylm-125m-20eps-seed964  (text, 125M, OPT, BabyLM)
    - znhoughton/opt-babylm-1.3b-20eps-seed964  (text, 1.3B, OPT, BabyLM)
    - allenai/OLMo-2-1124-7B       (text, 7B, OLMo-2, Dolma)
    - EleutherAI/pythia-6.9b       (text, 6.9B, GPT-NeoX, The Pile)
"""

import argparse
import csv
import gc
import io
import json
import logging
import os
import pickle
import queue
import random
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# HuggingFace memory / shard loading settings
# ---------------------------------------------------------------------------
os.environ.setdefault("HF_DATASETS_MAX_IN_MEMORY_SIZE", str(60 * 1024 ** 3))
os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", str(60 * 1024 ** 3))
os.environ.setdefault("DATASETS_VERBOSITY", "warning")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# HuggingFace fast tokenizers use Rust parallelism internally. When tokenization
# runs inside a Python thread (our prefetch_generator), the Rust threadpool can
# deadlock with Python's GIL. Setting this to "false" disables the internal Rust
# parallelism so the tokenizer is safe to call from background threads.
# The throughput loss is negligible — the GPU forward pass is the bottleneck.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from datasets import load_dataset, concatenate_datasets, Audio
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
    MimiModel,
    ParakeetForCTC,
    WhisperModel,
    WhisperProcessor,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("repr_analysis")


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    peak      = torch.cuda.max_memory_allocated() / 1024 ** 3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    tag = f"[{label}] " if label else ""
    logger.debug(
        f"{tag}GPU memory — "
        f"allocated: {allocated:.2f} GB  reserved: {reserved:.2f} GB  "
        f"peak: {peak:.2f} GB  total: {total:.2f} GB"
    )


def release_vram(label: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log_gpu_memory(f"after release [{label}]")


# ---------------------------------------------------------------------------
# Prefetch utility
# ---------------------------------------------------------------------------

def prefetch_generator(source_iter, queue_depth: int = 3):
    """
    Wraps any iterator and yields from a background thread.
    Keeps GPU busy by prefetching CPU-side preprocessing `queue_depth` batches ahead.
    """
    q = queue.Queue(maxsize=queue_depth)
    exc_holder = [None]

    def _producer():
        try:
            for item in source_iter:
                q.put(item)
        except Exception as e:
            exc_holder[0] = e
            logger.error(f"Prefetch producer thread crashed: {e}", exc_info=True)
        finally:
            q.put(None)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            if exc_holder[0] is not None:
                raise RuntimeError(f"Prefetch producer failed: {exc_holder[0]}") from exc_holder[0]
            break
        yield item


def _tokenized_batches(texts, tokenizer, start_batch, n_batches, batch_size, max_length):
    for batch_idx in range(start_batch, n_batches):
        i = batch_idx * batch_size
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        yield batch_idx, enc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = {
    # ── Audio encoders ────────────────────────────────────────────────────
    "whisper-base-enc": {
        "hf_id": "openai/whisper-base",
        "modality": "audio-whisper-enc",
        "params": "74M",
        "arch": "Whisper encoder",
        "corpus": "680k hrs audio",
    },
    "whisper-base-dec": {
        "hf_id": "openai/whisper-base",
        "modality": "audio-whisper-dec",
        "params": "74M",
        "arch": "Whisper decoder",
        "corpus": "680k hrs audio",
        # NOTE: The decoder is conditioned on both audio (cross-attention from the
        # encoder) and the previously generated tokens. We feed the encoder output
        # + a BOS token and pool the decoder's last hidden layer — this gives the
        # representation of "what the model predicts given the full audio context"
        # without requiring autoregressive decoding of the full transcript.
    },
    "parakeet-ctc-0.6b": {
        "hf_id": "nvidia/parakeet-ctc-0.6b",
        "modality": "audio-parakeet",
        "params": "600M",
        "arch": "FastConformer-CTC",
        "corpus": "Granary (64k hrs English)",
        # NOTE: CTC models have no autoregressive decoder — only encoder + linear
        # CTC projection. Decoder extraction is intentionally not supported.
    },
    "mimi": {
        "hf_id": "kyutai/mimi",
        "modality": "audio-mimi",
        "params": "~85M",
        "arch": "Conv+Transformer codec (EnCodec-style)",
        "corpus": "Speech data (Moshi training set)",
        # Mimi is a pure audio neural codec by Kyutai — no text training signal
        # whatsoever. It uses a streaming convolutional encoder-decoder with a
        # Transformer in both encoder and decoder, and a 16-codebook RVQ.
        # The first codebook is trained with WavLM distillation, giving it a
        # semantic component on top of acoustic reconstruction.
        # We extract the encoder's final continuous hidden state (the pre-RVQ
        # embedding), mean-pooled over time frames. This is the most "purely
        # acoustic" representation in the lineup — no language modelling objective.
        # Native 24kHz model; we resample LibriSpeech from 16kHz to 24kHz.
        # Natively supported in transformers via MimiModel.
    },
    # ── TTS / Omni / Speech-LM ───────────────────────────────────────────
    "qwen3-tts-1.7b": {
        "hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "modality": "tts-qwen3",
        "params": "1.7B",
        "arch": "Qwen3-TTS (discrete multi-codebook LM)",
        "corpus": "5M hrs speech, 10 languages",
        # Qwen3-TTS uses the custom `qwen-tts` package (not transformers).
        # Architecture: text tokenizer → Qwen3 LM backbone → speech codec tokens.
        # We feed the LibriSpeech transcript through the LM backbone and
        # mean-pool the last hidden layer over text token positions — this
        # captures the text representations that condition speech generation.
        # Install dependency: pip install qwen-tts
    },
    "voxtral-3b": {
        "hf_id": "mistralai/Voxtral-Mini-3B-2507",
        "modality": "audio-voxtral",
        "params": "3B",
        "arch": "Mistral+WhisperEnc",
        "corpus": "Mistral mix + speech",
        # Voxtral is Mistral's speech-understanding model. It consists of a
        # Whisper-based audio encoder + Mistral 3B LLM backbone. We extract the
        # LLM backbone's last hidden state after audio token projection.
    },
    "higgs-audio-v2-3b": {
        "hf_id": "bosonai/higgs-audio-v2-generation-3B-base",
        "modality": "tts-higgs",
        "params": "~5.8B",
        "arch": "Llama-3.2-3B + DualFFN adapter (2.2B)",
        "corpus": "AudioVerse (10M hrs speech+music+SFX)",
        # Higgs Audio V2 is a text-to-audio generation model built on Llama-3.2-3B.
        # The "generation variant" (this checkpoint) takes text as input and generates
        # discrete audio tokens as output — it is a TTS model, not audio-in/audio-out.
        # DualFFN adds parallel audio-specific FFN layers to each transformer block,
        # sharing attention layers between text and audio token streams.
        # We feed LibriSpeech transcripts (text) through the backbone and mean-pool
        # the last hidden layer — the same extraction strategy as Qwen3-TTS and the
        # text LLMs, since input is text in both cases.
        # Total params: ~5.8B (3B Llama backbone + 2.2B DualFFN adapter).
        # Transformers-native via HiggsAudioV2ForConditionalGeneration (no custom pkg).
    },
    # ── Text LLMs ─────────────────────────────────────────────────────────
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
}

ALL_SPLITS = [
    "train.100",
    "train.360",
    "train.500",
    "validation",
    "validation.other",
    "test",
    "test.other",
]

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
PCA_PLOT_MAX_PC = 10   # x-axis ceiling for PCA scatter and eigenspectra plots

MINIBATCH_SIZE = 2048
MINIBATCH_SEED = 42

N_STABILITY_SUBSETS = 10
STABILITY_SUBSET_FRAC = 0.8

# ── Modality groupings (used for heatmap separator line) ─────────────────
AUDIO_MODALITIES = {"audio-whisper-enc", "audio-whisper-dec", "audio-parakeet", "audio-mimi", "audio-voxtral", "tts-qwen3", "tts-higgs"}

MODEL_COLORS = {
    "whisper-base-enc":  "#1565C0",   # dark blue
    "whisper-base-dec":  "#42A5F5",   # light blue
    "parakeet-ctc-0.6b": "#00838F",   # teal
    "mimi":              "#D84315",   # deep orange-red
    "qwen3-tts-1.7b":    "#AD1457",   # deep pink
    "voxtral-3b":        "#FF6F00",   # amber
    "higgs-audio-v2-3b": "#558B2F",   # olive green
    "babylm-125m":       "#E65100",   # deep orange
    "babylm-1.3b":       "#F9A825",   # yellow
    "olmo-7b":           "#2E7D32",   # dark green
    "pythia-6.9b":       "#6A1B9A",   # purple
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

def load_librispeech(splits: list, max_samples: int | None,
                     cache_dir: Path = None, num_proc: int = 8):
    logger.info(f"{'='*60}")
    logger.info(f"Loading LibriSpeech splits: {splits}")
    logger.info(f"{'='*60}")

    cache_key = "_".join(sorted(splits)) + (f"_max{max_samples}" if max_samples else "")
    texts_cache_path = (cache_dir / f"texts_{cache_key}.json") if cache_dir else None

    parts = []
    for split in splits:
        if split not in SPLIT_CONFIG_MAP:
            raise ValueError(f"Unknown split '{split}'. Valid options: {list(SPLIT_CONFIG_MAP.keys())}")
        hf_config, hf_split = SPLIT_CONFIG_MAP[split]
        logger.info(f"  Fetching {split} (config={hf_config}, split={hf_split})…")

        for attempt in range(2):
            try:
                ds = load_dataset(
                    "openslr/librispeech_asr",
                    hf_config,
                    split=hf_split,
                    streaming=False,
                    trust_remote_code=True,
                    num_proc=num_proc,
                )
                break
            except Exception as e:
                # pyarrow.lib.ArrowInvalid means a shard file is corrupted.
                # Delete the HF cache for this dataset and retry once.
                if attempt == 0 and ("ArrowInvalid" in type(e).__name__ or
                                     "ArrowInvalid" in str(type(e).__mro__) or
                                     "null or length 0" in str(e) or
                                     "ipc" in str(e).lower()):
                    import shutil
                    hf_cache = Path(os.environ.get("HF_DATASETS_CACHE",
                                    Path.home() / ".cache" / "huggingface" / "datasets"))
                    corrupt_dirs = list(hf_cache.glob("openslr__librispeech*"))
                    if corrupt_dirs:
                        logger.warning(
                            f"Corrupted Arrow shard detected for split '{split}'. "
                            f"Deleting HF datasets cache and retrying download.\n"
                            f"  Removing: {[str(d) for d in corrupt_dirs]}"
                        )
                        for d in corrupt_dirs:
                            shutil.rmtree(d, ignore_errors=True)
                    else:
                        logger.warning(
                            f"ArrowInvalid error but could not find cache dir to clear. "
                            f"Try manually deleting: {hf_cache}/openslr__librispeech*"
                        )
                    logger.info(f"  Retrying {split} after cache clear…")
                else:
                    logger.error(f"Failed to load split '{split}': {e}")
                    raise
        else:
            raise RuntimeError(
                f"Failed to load split '{split}' after cache clear. "
                f"Check disk space and network connectivity."
            )

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

    if texts_cache_path and texts_cache_path.exists():
        logger.info(f"Loading cached transcripts from {texts_cache_path}…")
        with open(texts_cache_path) as f:
            texts = json.load(f)
    else:
        logger.info("Extracting transcripts (direct column access)…")
        # Use direct column access rather than dataset.map() — map() with
        # num_proc>1 forks worker processes that each serialize the full
        # dataset (audio arrays included), which OOMs on large splits.
        # String stripping is fast enough single-threaded.
        texts = [t.strip() for t in dataset["text"]]
        if texts_cache_path:
            with open(texts_cache_path, "w") as f:
                json.dump(texts, f)
            logger.info(f"Transcripts cached → {texts_cache_path}")

    # Disable automatic audio decoding so datasets does not invoke torchcodec
    # (which requires FFmpeg shared libs that are not available on the pod).
    # Audio bytes are decoded manually in _decode_audio_batch / the Mimi loop
    # using torchaudio, which works without system audio libraries.
    dataset = dataset.cast_column("audio", Audio(decode=False))
    logger.info(f"Final dataset size: {len(texts):,} utterances")
    return dataset, texts


# ---------------------------------------------------------------------------
# Embedding extraction — Whisper encoder
# ---------------------------------------------------------------------------

def extract_whisper_encoder_embeddings(
    model_id: str,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    checkpoint_dir: Path = None,
    prefetch_queue_depth: int = 3,
) -> np.ndarray:
    """Mean-pool Whisper encoder last hidden layer over time."""
    logger.info(f"Loading Whisper model (encoder): {model_id}")
    log_gpu_memory("before Whisper load")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    log_gpu_memory("after Whisper load")

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size
    checkpoint_path = checkpoint_dir / "whisper-base-enc_checkpoint.pkl" if checkpoint_dir else None

    start_batch, embeddings = _load_checkpoint(checkpoint_path, n_batches, "Whisper-enc")

    raw_iter = dataset.iter(batch_size=batch_size)
    if start_batch > 0:
        for _ in range(start_batch):
            next(raw_iter)
    batch_iter = prefetch_generator(raw_iter, queue_depth=prefetch_queue_depth)

    errors = 0
    for batch_idx in tqdm(range(start_batch, n_batches), desc="whisper-base-enc",
                          unit="batch", total=n_batches, initial=start_batch):
        batch = next(batch_iter)
        try:
            audio_arrays = _decode_audio_batch(batch)
            inputs = processor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            features = inputs.input_features.to(device, dtype=torch.float16)
            with torch.no_grad():
                # output_hidden_states=True returns a tuple of all layer outputs.
                # [-1] is the final encoder layer — consistent with all other extractors.
                enc_out = model.encoder(features, output_hidden_states=True)
                emb = enc_out.last_hidden_state.mean(dim=1).float().cpu().numpy()
            embeddings.append(emb)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _save_checkpoint(checkpoint_path, embeddings, batch_idx)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        _maybe_save_checkpoint(checkpoint_path, embeddings, batch_idx, n_batches)

    if errors:
        logger.warning(f"whisper-enc: skipped {errors} batches")
    del model
    release_vram("whisper-enc")
    _remove_checkpoint(checkpoint_path)
    if not embeddings:
        raise RuntimeError("All batches failed — no embeddings were collected.")
    result = np.concatenate(embeddings, axis=0)
    logger.info(f"Whisper-enc embeddings shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Embedding extraction — Whisper decoder
# ---------------------------------------------------------------------------

def extract_whisper_decoder_embeddings(
    model_id: str,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    checkpoint_dir: Path = None,
    prefetch_queue_depth: int = 3,
) -> np.ndarray:
    """
    Extract Whisper *decoder* representations.

    Strategy: run the encoder forward pass to get encoder_hidden_states, then
    run the decoder for a single step with just the BOS (start-of-transcript)
    token. Pool the decoder last hidden layer over the single time step.

    This gives a representation of "what the decoder sees given the full audio
    context" without requiring autoregressive transcript decoding, which would
    be ~10–100× slower and would conflate representations across tokens.
    """
    logger.info(f"Loading Whisper model (decoder): {model_id}")
    log_gpu_memory("before Whisper-dec load")

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()

    # BOS token id for the decoder's first-step input
    bos_token_id = model.config.decoder_start_token_id
    log_gpu_memory("after Whisper-dec load")

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size
    checkpoint_path = checkpoint_dir / "whisper-base-dec_checkpoint.pkl" if checkpoint_dir else None

    start_batch, embeddings = _load_checkpoint(checkpoint_path, n_batches, "Whisper-dec")

    raw_iter = dataset.iter(batch_size=batch_size)
    if start_batch > 0:
        for _ in range(start_batch):
            next(raw_iter)
    batch_iter = prefetch_generator(raw_iter, queue_depth=prefetch_queue_depth)

    errors = 0
    for batch_idx in tqdm(range(start_batch, n_batches), desc="whisper-base-dec",
                          unit="batch", total=n_batches, initial=start_batch):
        batch = next(batch_iter)
        try:
            audio_arrays = _decode_audio_batch(batch)
            inputs = processor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            features = inputs.input_features.to(device, dtype=torch.float16)

            B = features.shape[0]
            # Single BOS token as the only decoder input — shape (B, 1)
            decoder_input_ids = torch.full(
                (B, 1), bos_token_id, dtype=torch.long, device=device
            )

            with torch.no_grad():
                # Step 1: run the encoder to get cross-attention key/value states.
                enc_out = model.encoder(features, output_hidden_states=False)

                # Step 2: run the decoder for a single BOS step, explicitly requesting
                # hidden states. Calling model.decoder() directly guarantees
                # output_hidden_states is applied to the decoder, not the encoder —
                # the top-level WhisperModel.forward() does not reliably pass the flag
                # through in all transformers versions.
                dec_out = model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=enc_out.last_hidden_state,
                    output_hidden_states=True,
                )
                # hidden_states: tuple of (n_decoder_layers+1) tensors, each (B, T_dec, D).
                # T_dec = 1 (single BOS token), so mean over dim=1 == squeeze.
                dec_hidden = dec_out.hidden_states[-1]            # (B, 1, D)
                emb = dec_hidden.mean(dim=1).float().cpu().numpy()  # (B, D)

            embeddings.append(emb)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _save_checkpoint(checkpoint_path, embeddings, batch_idx)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        _maybe_save_checkpoint(checkpoint_path, embeddings, batch_idx, n_batches)

    if errors:
        logger.warning(f"whisper-dec: skipped {errors} batches")
    del model
    release_vram("whisper-dec")
    _remove_checkpoint(checkpoint_path)
    if not embeddings:
        raise RuntimeError("All batches failed — no embeddings were collected.")
    result = np.concatenate(embeddings, axis=0)
    logger.info(f"Whisper-dec embeddings shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Embedding extraction — Parakeet CTC (encoder only — no decoder exists)
# ---------------------------------------------------------------------------

def extract_parakeet_embeddings(
    model_id: str,
    dataset,
    device: torch.device,
    batch_size: int = 32,
    checkpoint_dir: Path = None,
    prefetch_queue_depth: int = 3,
) -> np.ndarray:
    """
    Batched extraction of Parakeet CTC encoder embeddings.

    IMPORTANT: Parakeet CTC is an encoder-only CTC model. It has no autoregressive
    decoder — only encoder + linear CTC projection. Decoder extraction is not
    possible/meaningful for this architecture.
    """
    logger.info(f"Loading Parakeet model: {model_id}")
    log_gpu_memory("before Parakeet load")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    # Parakeet uses the NeMo framework internally. Loading with device_map="auto"
    # triggers "Invalid device argument: did you call init?" because NeMo's CUDA
    # init hasn't run. Load to CPU first, then move to device explicitly.
    model = ParakeetForCTC.from_pretrained(model_id, torch_dtype=torch.float32)
    model = model.to(device).eval()

    first_device = device
    log_gpu_memory("after Parakeet load")

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size
    checkpoint_path = checkpoint_dir / "parakeet-ctc-0.6b_checkpoint.pkl" if checkpoint_dir else None

    start_batch, embeddings = _load_checkpoint(checkpoint_path, n_batches, "Parakeet")

    raw_iter = dataset.iter(batch_size=batch_size)
    if start_batch > 0:
        for _ in range(start_batch):
            next(raw_iter)
    batch_iter = prefetch_generator(raw_iter, queue_depth=prefetch_queue_depth)

    errors = 0
    for batch_idx in tqdm(range(start_batch, n_batches), desc="parakeet-ctc-0.6b",
                          unit="batch", total=n_batches, initial=start_batch):
        batch = next(batch_iter)
        try:
            audio_arrays = _decode_audio_batch(batch)
            inputs = feature_extractor(
                audio_arrays, sampling_rate=SAMPLE_RATE,
                return_tensors="pt", padding="longest",
            )
            input_key = next(k for k in inputs.keys() if "mask" not in k)
            input_tensor = inputs[input_key].to(first_device, dtype=torch.float32)
            # Extract attention_mask if present (it's created by padding="longest")
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(first_device)
            with torch.no_grad():
                # Call the FastConformer encoder directly rather than the full
                # ParakeetForCTC wrapper. The CTC wrapper's hidden_states[-1] can
                # include the linear CTC projection layer in some transformers
                # versions, which is not a meaningful representation layer.
                # model.parakeet is the FastConformer backbone; .encoder is the
                # conformer stack. We take its final hidden state directly.
                if hasattr(model, "parakeet") and hasattr(model.parakeet, "encoder"):
                    enc_out = model.parakeet.encoder(
                        input_tensor, output_hidden_states=True
                    )
                    last_hidden = enc_out.last_hidden_state
                else:
                    # Fallback: use the full model but take hidden_states from
                    # the output, which excludes the CTC head in most versions.
                    out = model(input_tensor, output_hidden_states=True)
                    last_hidden = out.hidden_states[-1]
                # Masked mean-pool: exclude padding frames from the average.
                # Parakeet's attention_mask is over input frames (pre-subsampling);
                # last_hidden is post-subsampling so lengths differ. Fall back to
                # unmasked mean if shapes don't align — padding is minimal at
                # batch_size=32 with padding="longest".
                if attention_mask is not None and attention_mask.shape[1] == last_hidden.shape[1]:
                    mask = attention_mask.float().unsqueeze(-1)
                    emb = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    emb = emb.float().cpu().numpy()
                else:
                    emb = last_hidden.mean(dim=1).float().cpu().numpy()
            if not np.isfinite(emb).all():
                logger.warning(f"Batch {batch_idx}: non-finite Parakeet output, skipping")
                errors += 1
                continue
            embeddings.append(emb)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _save_checkpoint(checkpoint_path, embeddings, batch_idx)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        _maybe_save_checkpoint(checkpoint_path, embeddings, batch_idx, n_batches)

    if errors:
        logger.warning(f"Parakeet: skipped {errors} batches")
    del model
    release_vram("parakeet")
    _remove_checkpoint(checkpoint_path)
    if not embeddings:
        raise RuntimeError("All batches failed — no embeddings were collected.")
    result = np.concatenate(embeddings, axis=0)
    logger.info(f"Parakeet embeddings shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Embedding extraction — Mimi (pure audio codec encoder, Kyutai)
# ---------------------------------------------------------------------------

MIMI_SAMPLE_RATE = 24_000   # Mimi's native sample rate; LibriSpeech is 16kHz


def extract_mimi_embeddings(
    model_id: str,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    checkpoint_dir: Path = None,
    prefetch_queue_depth: int = 3,
) -> np.ndarray:
    """
    Extract Mimi codec encoder pre-RVQ continuous hidden states.

    Architecture:
        raw 24kHz audio
          → causal convolutional encoder (downsamples to 12.5 Hz frames)
          → Transformer encoder (adds context across frames)
          → pre-RVQ continuous embedding  ← we extract this
          → residual vector quantizer (16 codebooks → discrete tokens)
          → Transformer decoder + convolutional decoder → reconstructed audio

    The pre-RVQ embedding is Mimi's continuous latent — the richest
    representation before information is discarded by quantization.
    It carries both semantic (via WavLM distillation on codebook 1) and
    acoustic content, making it the ideal single representation to compare.

    We call model.encode() which returns audio_codes (discrete) but also
    exposes encoder_last_hidden_state via output_hidden_states on the
    encoder submodule. Alternatively we call model.encoder directly to
    get the continuous pre-RVQ hidden state unambiguously.

    Resampling: LibriSpeech is 16kHz; Mimi expects 24kHz. We use
    torchaudio.functional.resample for correct anti-aliased resampling
    (unlike the naive integer-decimation used for same-rate models).

    Input:  raw 16kHz audio waveforms from LibriSpeech
    Output: mean-pooled pre-RVQ encoder hidden state (B, D=512)
    """
    logger.info(f"Loading Mimi model: {model_id}")
    log_gpu_memory("before Mimi load")

    try:
        import torchaudio
    except ImportError:
        raise ImportError("torchaudio required for Mimi resampling: pip install torchaudio")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = MimiModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model = model.to(device).eval()
    log_gpu_memory("after Mimi load")

    # Verify the feature extractor's expected sample rate matches MIMI_SAMPLE_RATE
    fe_sr = getattr(feature_extractor, "sampling_rate", MIMI_SAMPLE_RATE)
    if fe_sr != MIMI_SAMPLE_RATE:
        logger.warning(
            f"Mimi feature extractor sampling_rate={fe_sr} differs from "
            f"expected {MIMI_SAMPLE_RATE}. Using feature extractor's value."
        )
    target_sr = fe_sr

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size
    checkpoint_path = checkpoint_dir / "mimi_checkpoint.pkl" if checkpoint_dir else None

    start_batch, embeddings = _load_checkpoint(checkpoint_path, n_batches, "Mimi")

    raw_iter = dataset.iter(batch_size=batch_size)
    if start_batch > 0:
        for _ in range(start_batch):
            next(raw_iter)
    batch_iter = prefetch_generator(raw_iter, queue_depth=prefetch_queue_depth)

    errors = 0
    for batch_idx in tqdm(range(start_batch, n_batches), desc="mimi",
                          unit="batch", total=n_batches, initial=start_batch):
        batch = next(batch_iter)
        try:
            # Decode audio and resample 16kHz → 24kHz using torchaudio.
            # Handles both decoded {"array":...} and undecoded {"bytes":...}
            # formats (the latter when dataset uses Audio(decode=False)).
            audio_arrays_24k = []
            for audio_item in batch["audio"]:
                if isinstance(audio_item, dict) and "array" in audio_item:
                    arr = np.array(audio_item["array"], dtype=np.float32)
                    sr  = audio_item.get("sampling_rate", SAMPLE_RATE)
                    audio = torch.from_numpy(arr).unsqueeze(0)  # (1, T)
                elif isinstance(audio_item, dict):
                    raw_bytes = audio_item.get("bytes")
                    path      = audio_item.get("path")
                    if raw_bytes:
                        audio, sr = torchaudio.load(io.BytesIO(raw_bytes))
                    elif path:
                        audio, sr = torchaudio.load(path)
                    else:
                        raise ValueError(f"Audio item has neither bytes nor path: {list(audio_item.keys())}")
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)  # stereo → mono
                else:
                    audio = torch.from_numpy(np.array(audio_item, dtype=np.float32)).unsqueeze(0)
                    sr = SAMPLE_RATE
                if sr != target_sr:
                    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
                # Truncate to MAX_AUDIO_SECONDS
                max_samples = MAX_AUDIO_SECONDS * target_sr
                audio = audio[:, :max_samples]
                audio_arrays_24k.append(audio.squeeze(0).numpy())

            # feature_extractor pads to longest in batch and returns padding_mask
            inputs = feature_extractor(
                raw_audio=audio_arrays_24k,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(device)
            # padding_mask: 1 = real frame, 0 = padding (Mimi convention)
            padding_mask = inputs.get("padding_mask", None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            with torch.no_grad():
                # Run the encoder submodule directly to get continuous pre-RVQ
                # hidden states. MimiModel.encoder returns a BaseModelOutput
                # whose last_hidden_state is the pre-quantization embedding.
                enc_out = model.encoder(
                    input_values,
                    padding_mask=padding_mask,
                )
                last_hidden = enc_out.last_hidden_state   # (B, T_frames, D)

            # Mean-pool over time frames, excluding padding.
            # padding_mask is at the *input* sample level, not the frame level —
            # the encoder downsamples by ~1920x (24000/12.5), so we can't apply
            # it directly. Use unmasked mean; padding is a tiny fraction at
            # batch_size=64 with variable-length audio.
            emb = last_hidden.float().mean(dim=1).cpu().numpy()   # (B, D)

            if not np.isfinite(emb).all():
                n_bad = (~np.isfinite(emb)).sum()
                logger.warning(f"Batch {batch_idx}: {n_bad} non-finite Mimi values, skipping")
                errors += 1
                continue

            embeddings.append(emb)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"CUDA OOM at mimi batch {batch_idx}. "
                    f"Try --mimi_batch_size < {batch_size}."
                )
                _save_checkpoint(checkpoint_path, embeddings, batch_idx)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        _maybe_save_checkpoint(checkpoint_path, embeddings, batch_idx, n_batches)

    if errors:
        logger.warning(f"Mimi: skipped {errors} batches")
    del model
    release_vram("mimi")
    _remove_checkpoint(checkpoint_path)
    if not embeddings:
        raise RuntimeError("All batches failed — no embeddings were collected.")
    result = np.concatenate(embeddings, axis=0)
    logger.info(f"Mimi embeddings shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Embedding extraction — Omni / multimodal audio-LLM (Qwen2.5-Omni, Voxtral)
# ---------------------------------------------------------------------------

def extract_voxtral_embeddings(
    model_name: str,
    model_id: str,
    dataset,
    device: torch.device,
    batch_size: int = 8,
    checkpoint_dir: Path = None,
    prefetch_queue_depth: int = 3,
) -> np.ndarray:
    """
    Extract the LLM backbone's last hidden state for Voxtral.

    Voxtral's architecture:
      1. A Whisper-style audio encoder encodes audio frames → audio tokens
      2. A projection layer maps audio tokens into the LLM's embedding space
      3. The Mistral 3B LLM backbone processes the audio token sequence

    We pool the LLM backbone's last hidden layer over all token positions.
    This captures how the language backbone organises audio-derived representations.

    Batching note: batch_size defaults to 8 because Voxtral audio sequences are
    long. Reduce with --voxtral_batch_size if you hit OOM.
    """
    logger.info(f"Loading Voxtral model: {model_id}  (label={model_name})")
    log_gpu_memory(f"before {model_name} load")

    # device_map="auto" conflicts with NeMo's CUDA init (from Parakeet) and
    # breaks all subsequent model loads. Load to CPU then move explicitly.
    load_kwargs = dict(
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)

    model = model.to(device).eval()
    first_device = device
    log_gpu_memory(f"after {model_name} load")

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size
    checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint.pkl" if checkpoint_dir else None

    start_batch, embeddings = _load_checkpoint(checkpoint_path, n_batches, model_name)

    raw_iter = dataset.iter(batch_size=batch_size)
    if start_batch > 0:
        for _ in range(start_batch):
            next(raw_iter)
    batch_iter = prefetch_generator(raw_iter, queue_depth=prefetch_queue_depth)

    errors = 0
    for batch_idx in tqdm(range(start_batch, n_batches), desc=model_name,
                          unit="batch", total=n_batches, initial=start_batch):
        batch = next(batch_iter)
        try:
            audio_arrays = _decode_audio_batch(batch)

            # Voxtral's processor wraps WhisperFeatureExtractor internally,
            # which requires the audio under the key 'raw_speech'.
            try:
                inputs = processor(
                    raw_speech=audio_arrays,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )
            except TypeError:
                # Fallback: pass positionally in case the outer processor
                # dispatches differently.
                inputs = processor(
                    audio_arrays,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )

            # Move all tensor inputs to the first-layer device
            inputs = {
                k: v.to(first_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

                # Extract last hidden state inside no_grad.
                # For CausalLM outputs, hidden_states is the only valid attribute.
                # encoder_last_hidden_state and last_hidden_state don't exist on
                # CausalLMOutputWithPast — accessing them raises AttributeError.
                if hasattr(out, "hidden_states") and out.hidden_states is not None:
                    last_hidden = out.hidden_states[-1]   # (B, T, D)
                else:
                    raise ValueError(
                        f"{model_name}: output_hidden_states=True was set but "
                        f"out.hidden_states is None. Check that the model class "
                        f"supports output_hidden_states."
                    )

            emb = last_hidden.float().mean(dim=1).cpu().numpy()  # (B, D)

            if not np.isfinite(emb).all():
                logger.warning(f"Batch {batch_idx}: non-finite {model_name} output, skipping")
                errors += 1
                continue

            embeddings.append(emb)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"CUDA OOM at {model_name} batch {batch_idx}. "
                    f"Try --voxtral_batch_size < {batch_size}."
                )
                _save_checkpoint(checkpoint_path, embeddings, batch_idx)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        _maybe_save_checkpoint(checkpoint_path, embeddings, batch_idx, n_batches)

    if errors:
        logger.warning(f"{model_name}: skipped {errors} batches")
    del model
    release_vram(model_name)
    _remove_checkpoint(checkpoint_path)
    if not embeddings:
        raise RuntimeError("All batches failed — no embeddings were collected.")
    result = np.concatenate(embeddings, axis=0)
    logger.info(f"{model_name} embeddings shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Embedding extraction — Higgs Audio V2 (TTS: text → audio tokens)
# ---------------------------------------------------------------------------

def extract_higgs_audio_embeddings(
    model_id: str,
    texts: list,
    device: torch.device,
    batch_size: int = 16,
    checkpoint_dir: Path = None,
    prefetch_queue_depth: int = 3,
) -> np.ndarray:
    """
    Extract Higgs Audio V2 LLM backbone hidden states on text input.

    Architecture clarification:
        Higgs Audio V2 (generation variant) is a TEXT-TO-AUDIO model:
            text tokens → Llama-3.2-3B backbone + DualFFN audio adapter
                        → discrete audio codec tokens (25fps)

        The "3B" checkpoint is the generation variant and takes TEXT as input.
        The understanding variant (audio input) exists but is not open-sourced.

        Total parameters: ~5.8B (3B Llama backbone + 2.2B DualFFN adapter).

    Extraction strategy:
        Feed LibriSpeech transcripts through the backbone via the standard
        HiggsAudioV2ForConditionalGeneration forward pass with text-only input
        and mean-pool the last hidden layer over non-padding token positions.
        This is directly comparable to Qwen3-TTS and the text LLMs — all three
        process text, letting us ask: does TTS training (on audio generation)
        change how the LM backbone represents text?

    Requires: pip install --upgrade transformers  (added 2026-02-19, needs >=4.50)
    """
    try:
        from transformers import HiggsAudioV2ForConditionalGeneration
    except ImportError:
        raise ImportError(
            "HiggsAudioV2ForConditionalGeneration not found. "
            "Upgrade: pip install --upgrade transformers  (needs >=4.50, added 2026-02-19)."
        )

    logger.info(f"Loading Higgs Audio V2 model: {model_id}")
    log_gpu_memory("before Higgs Audio V2 load")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device).eval()

    # The LLM backbone sits at model.model (Llama-3.2-3B with DualFFN layers).
    # Tokenizer is accessible via processor.tokenizer.
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    first_device = device
    logger.info(f"Higgs Audio V2 first layer device: {first_device}")
    log_gpu_memory("after Higgs Audio V2 load")

    label = "higgs-audio-v2-3b"
    n = len(texts)
    n_batches = (n + batch_size - 1) // batch_size
    checkpoint_path = checkpoint_dir / f"{label}_checkpoint.pkl" if checkpoint_dir else None

    start_batch, embeddings = _load_checkpoint(checkpoint_path, n_batches, label)

    # Tokenise in background thread (same pattern as text LLMs)
    token_iter = prefetch_generator(
        _tokenized_batches(texts, tokenizer, start_batch, n_batches, batch_size, MAX_TEXT_TOKENS),
        queue_depth=prefetch_queue_depth,
    )

    errors = 0
    for batch_idx, enc in tqdm(token_iter, desc=label,
                                unit="batch", total=n_batches - start_batch):
        try:
            input_ids = enc["input_ids"].to(first_device)
            attention_mask = enc["attention_mask"].to(first_device)

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # hidden_states: tuple of (n_layers+1) tensors, each (B, T, D).
                # [-1] is the final transformer block output.
                hidden = out.hidden_states[-1].float()   # (B, T, D)

            # Masked mean-pool over non-padding tokens (consistent with text LLMs)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = pooled.cpu().numpy()

            if not np.isfinite(emb).all():
                n_bad = (~np.isfinite(emb)).sum()
                logger.warning(f"Batch {batch_idx}: {n_bad} non-finite Higgs values, skipping")
                errors += 1
                continue

            embeddings.append(emb)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"CUDA OOM at {label} batch {batch_idx}. "
                    f"Try --higgs_batch_size < {batch_size}."
                )
                _save_checkpoint(checkpoint_path, embeddings, batch_idx)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        _maybe_save_checkpoint(checkpoint_path, embeddings, batch_idx, n_batches)

    if errors:
        logger.warning(f"{label}: skipped {errors} batches")
    del model
    release_vram(label)
    _remove_checkpoint(checkpoint_path)
    if not embeddings:
        raise RuntimeError("All batches failed — no embeddings were collected.")
    result = np.concatenate(embeddings, axis=0)
    logger.info(f"{label} embeddings shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Embedding extraction — Qwen3-TTS
# ---------------------------------------------------------------------------

def extract_qwen3_tts_embeddings(
    model_id: str,
    texts: list,
    device: torch.device,
    batch_size: int = 32,
    checkpoint_dir: Path = None,
    prefetch_queue_depth: int = 3,
) -> np.ndarray:
    """
    Extract Qwen3-TTS LM backbone hidden states on input text.

    Qwen3-TTS is a text-to-speech model with architecture:
        text tokens → Qwen3 LM backbone → speech codec token predictions

    Since LibriSpeech gives us transcripts (text), not target audio, the
    natural extraction point is the LM backbone processing the input text —
    identical in spirit to how we extract from text LLMs. This lets us ask:
    "how similar are the text representations a TTS LM forms to those of
    a pure text LLM?" — a direct Platonic Representation Hypothesis test
    across TTS vs. pure-language training objectives.

    The qwen-tts package wraps a Qwen3 transformer. We access its underlying
    language model via model.language_model (or model.model, depending on
    version), then tokenize with its bundled tokenizer and extract the last
    hidden layer with mean pooling over non-padding tokens.

    Requires: pip install qwen-tts
    """
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        raise ImportError(
            "qwen-tts package not found. Install with: pip install qwen-tts\n"
            "Then re-run without --skip_extraction to extract Qwen3-TTS embeddings."
        )

    logger.info(f"Loading Qwen3-TTS model: {model_id}")
    log_gpu_memory("before Qwen3-TTS load")

    model = Qwen3TTSModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    # Qwen3TTSModel is not a standard nn.Module in all versions — to/eval may not exist.
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    # Qwen3-TTS wraps a Qwen3 transformer. The LM backbone is accessible via
    # .language_model (most versions) or .model. We need it for hidden states.
    if hasattr(model, "language_model"):
        lm = model.language_model
    elif hasattr(model, "model"):
        lm = model.model
    else:
        raise AttributeError(
            f"Cannot find LM backbone in Qwen3TTSModel. "
            f"Available attributes: {[a for a in dir(model) if not a.startswith('_')]}\n"
            f"Please open an issue or check the qwen-tts version."
        )

    # The tokenizer is bundled with the model. Try common attribute names.
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    elif hasattr(model, "text_tokenizer"):
        tokenizer = model.text_tokenizer
    else:
        # Fall back to loading from HF hub with the same repo
        logger.warning("Tokenizer not found on model object — loading via AutoTokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine the device of the LM backbone's first layer
    if hasattr(lm, "hf_device_map") and lm.hf_device_map:
        first_device = next(iter(lm.hf_device_map.values()))
    else:
        first_device = next(lm.parameters()).device

    logger.info(f"Qwen3-TTS LM backbone first device: {first_device}")
    log_gpu_memory("after Qwen3-TTS load")

    n = len(texts)
    n_batches = (n + batch_size - 1) // batch_size
    label = "qwen3-tts-1.7b"
    checkpoint_path = checkpoint_dir / f"{label}_checkpoint.pkl" if checkpoint_dir else None

    start_batch, embeddings = _load_checkpoint(checkpoint_path, n_batches, label)

    token_iter = prefetch_generator(
        _tokenized_batches(texts, tokenizer, start_batch, n_batches, batch_size, MAX_TEXT_TOKENS),
        queue_depth=prefetch_queue_depth,
    )

    errors = 0
    for batch_idx, enc in tqdm(token_iter, desc=label,
                                unit="batch", total=n_batches - start_batch):
        try:
            input_ids = enc["input_ids"].to(first_device)
            attention_mask = enc["attention_mask"].to(first_device)
            with torch.no_grad():
                out = lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1].float()   # (B, T, D)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"CUDA OOM at {label} batch {batch_idx}. "
                    f"Try reducing --lm_batch_size (currently {batch_size})."
                )
                _save_checkpoint(checkpoint_path, embeddings, batch_idx)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        _maybe_save_checkpoint(checkpoint_path, embeddings, batch_idx, n_batches)

    if errors:
        logger.warning(f"{label}: skipped {errors} batches")
    # Guard: lm may be unbound if the AttributeError branch was hit before
    # assignment. Always delete model; delete lm only if it was assigned.
    del model
    if "lm" in locals():
        del lm
    release_vram(label)
    _remove_checkpoint(checkpoint_path)
    if not embeddings:
        raise RuntimeError("All batches failed — no embeddings were collected.")
    result = np.concatenate(embeddings, axis=0)
    logger.info(f"{label} embeddings shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Embedding extraction — text LLMs
# ---------------------------------------------------------------------------

def extract_lm_embeddings(
    model_id: str,
    texts: list,
    device: torch.device,
    batch_size: int = 32,
    checkpoint_dir: Path = None,
    prefetch_queue_depth: int = 3,
) -> np.ndarray:
    """Mean-pool last hidden layer over non-padding tokens."""
    logger.info(f"Loading LM: {model_id}")
    log_gpu_memory(f"before {model_id.split('/')[-1]} load")

    load_kwargs = dict(
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)
    model = model.to(device).eval()

    first_device = device

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n = len(texts)
    n_batches = (n + batch_size - 1) // batch_size
    label = model_id.split("/")[-1]
    checkpoint_path = checkpoint_dir / f"{label}_checkpoint.pkl" if checkpoint_dir else None

    start_batch, embeddings = _load_checkpoint(checkpoint_path, n_batches, label)
    log_gpu_memory(f"after {label} load")

    token_iter = prefetch_generator(
        _tokenized_batches(texts, tokenizer, start_batch, n_batches, batch_size, MAX_TEXT_TOKENS),
        queue_depth=prefetch_queue_depth,
    )

    errors = 0
    for batch_idx, enc in tqdm(token_iter, desc=label,
                                unit="batch", total=n_batches - start_batch):
        try:
            input_ids = enc["input_ids"].to(first_device)
            attention_mask = enc["attention_mask"].to(first_device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
                hidden = out.hidden_states[-1].float()
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _save_checkpoint(checkpoint_path, embeddings, batch_idx)
                torch.cuda.empty_cache()
                raise
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch {batch_idx}: {e}")
        _maybe_save_checkpoint(checkpoint_path, embeddings, batch_idx, n_batches)

    if errors:
        logger.warning(f"{label}: skipped {errors} batches")
    del model
    release_vram(label)
    _remove_checkpoint(checkpoint_path)
    if not embeddings:
        raise RuntimeError("All batches failed — no embeddings were collected.")
    result = np.concatenate(embeddings, axis=0)
    logger.info(f"{label} embeddings shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers (DRY)
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY = 500


def _load_checkpoint(path, n_batches, label):
    """Return (start_batch, partial_embeddings_list). Loads .pkl if it exists."""
    if path and path.exists():
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        embeddings = ckpt["embeddings"]
        start_batch = ckpt["next_batch"]
        logger.info(f"Resuming {label} from batch {start_batch}/{n_batches}")
        return start_batch, embeddings
    return 0, []


def _save_checkpoint(path, embeddings, next_batch):
    if path:
        with open(path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "next_batch": next_batch}, f)


def _maybe_save_checkpoint(path, embeddings, batch_idx, n_batches):
    if path and (batch_idx + 1) % CHECKPOINT_EVERY == 0:
        _save_checkpoint(path, embeddings, batch_idx + 1)
        logger.debug(f"Checkpoint at {batch_idx+1}/{n_batches}")


def _remove_checkpoint(path):
    if path and path.exists():
        path.unlink()
        logger.debug(f"Checkpoint removed: {path.name}")


def _decode_audio_batch(batch) -> list:
    """Decode a HuggingFace audio batch dict into a list of float32 numpy arrays.

    Handles both the decoded format {"array": ..., "sampling_rate": ...} and the
    undecoded format {"bytes": ..., "path": ...} produced by Audio(decode=False).
    We use decode=False on the dataset to avoid torchcodec (which requires FFmpeg
    shared libs that are not available on the pod). torchaudio decodes FLAC/WAV
    from bytes or path without needing system audio libraries.

    Resampling note: the integer decimation `audio[::ratio]` is only exact when
    the source sample rate is an integer multiple of SAMPLE_RATE (16kHz).
    LibriSpeech is always 16kHz so this is fine in practice.
    """
    import torchaudio

    audio_arrays = []
    for audio_item in batch["audio"]:
        if isinstance(audio_item, dict) and "array" in audio_item:
            # Already-decoded path (legacy / if cast_column wasn't applied)
            arr = np.array(audio_item["array"], dtype=np.float32)
            sr  = audio_item.get("sampling_rate", SAMPLE_RATE)
        elif isinstance(audio_item, dict):
            # decode=False path: {"bytes": b"...", "path": "..."}
            raw_bytes = audio_item.get("bytes")
            path      = audio_item.get("path")
            if raw_bytes:
                t, sr = torchaudio.load(io.BytesIO(raw_bytes))
            elif path:
                t, sr = torchaudio.load(path)
            else:
                raise ValueError(f"Audio item has neither bytes nor path: {list(audio_item.keys())}")
            if t.shape[0] > 1:
                t = t.mean(dim=0, keepdim=True)  # stereo → mono
            arr = t.squeeze(0).numpy().astype(np.float32)
        else:
            arr = np.array(audio_item, dtype=np.float32)
            sr  = SAMPLE_RATE

        audio = arr
        if sr != SAMPLE_RATE:
            audio = audio[:: int(sr / SAMPLE_RATE)]
        audio = audio[: MAX_AUDIO_SECONDS * SAMPLE_RATE]
        audio_arrays.append(audio)
    return audio_arrays


# ---------------------------------------------------------------------------
# Minibatch CKA  (Nguyen, Raghu & Kornblith 2021)
# ---------------------------------------------------------------------------

def _hsic1_batch(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased HSIC_1 estimator (Song et al. 2007 U-statistic)."""
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
    """Minibatch linear CKA (Nguyen, Raghu & Kornblith 2021)."""
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
        Xb = X.astype(np.float64)
        Yb = Y.astype(np.float64)
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-10
        Yb /= np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-10
        denom = np.sqrt(max(_hsic1_batch(Xb, Xb), 0.0) * max(_hsic1_batch(Yb, Yb), 0.0))
        return float(_hsic1_batch(Xb, Yb) / denom) if denom > 1e-10 else 0.0

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
    """
    Compute the top-n_components normalised eigenvalues of the covariance matrix.

    Uses randomised SVD (Halko et al. 2011) rather than full SVD. On a 280k × 4096
    matrix, full SVD would allocate O(N²) memory (~600 GB) and never complete.
    Randomised SVD runs in O(N × n_components) memory and takes seconds.
    """
    from sklearn.utils.extmath import randomized_svd

    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    if not np.isfinite(Xc).all():
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    Xc_scaled = Xc / np.sqrt(n - 1)

    # n_oversamples adds stability to the randomised approximation.
    # n_iter=4 gives tight singular value estimates with minimal extra passes.
    _, s, _ = randomized_svd(
        Xc_scaled,
        n_components=n_components,
        n_oversamples=10,
        n_iter=4,
        random_state=MINIBATCH_SEED,
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

def plot_eigenspectra(eigenvalues: dict, plots_dir: Path,
                      max_pc: int = PCA_PLOT_MAX_PC):
    """
    Individual eigenspectra panels, x-axis limited to the first `max_pc`
    principal components (default = 10).
    """
    _apply_style()
    names = list(eigenvalues.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        eigs = eigenvalues[name][:max_pc]   # ← truncate to max_pc
        color = MODEL_COLORS.get(name, "#555555")
        k = np.arange(1, len(eigs) + 1)
        ax.fill_between(k, eigs, alpha=0.18, color=color)
        ax.plot(k, eigs, color=color, linewidth=2.2)
        er = effective_rank(eigenvalues[name])   # still computed on all components
        ax.set_title(f"{name}\n(eff. rank = {er:.1f})", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Principal Component (1–{max_pc})", fontsize=9)
        ax.set_ylabel("Fraction of Variance", fontsize=9)
        ax.set_xlim(1, max_pc)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    fig.suptitle(
        f"PCA Eigenvalue Spectra  ·  first {max_pc} PCs  ·  similar decay → similar intrinsic dimensionality",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = plots_dir / "eigenspectra.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_eigenspectra_overlay(eigenvalues: dict, plots_dir: Path,
                               max_pc: int = PCA_PLOT_MAX_PC):
    """
    All eigenspectra overlaid on one axes, x-axis limited to `max_pc`.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, eigs_full in eigenvalues.items():
        eigs = eigs_full[:max_pc]           # ← truncate to max_pc
        color = MODEL_COLORS.get(name, "#555555")
        k = np.arange(1, len(eigs) + 1)
        ax.plot(k, eigs, color=color, linewidth=2, label=name, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlim(1, max_pc)
    ax.set_xlabel(f"Principal Component (1–{max_pc})", fontsize=11)
    ax.set_ylabel("Fraction of Variance (log)", fontsize=11)
    ax.set_title(f"Eigenvalue Spectra — All Models Overlaid  (first {max_pc} PCs)",
                 fontsize=13, fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.2), 4))
    bars = ax.bar(names, ranks, color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{rank:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Effective Rank", fontsize=11)
    ax.set_title("Effective Rank of Representation Spaces\n"
                 "Higher = variance spread across more dimensions",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9, rotation=20)
    plt.tight_layout()
    path = plots_dir / "effective_rank.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_pca_scatter(embeddings: dict, plots_dir: Path,
                     max_scatter_samples: int = 10_000):
    """
    2D PCA scatter plots.

    Subsamples to max_scatter_samples before fitting PCA — fitting sklearn PCA
    on a 280k × 4096 matrix requires O(N²) memory and will OOM. 10k points is
    more than sufficient to show the geometry and renders quickly.
    """
    from sklearn.decomposition import PCA as skPCA
    _apply_style()

    clean = {k: v for k, v in embeddings.items() if np.isfinite(v).all()}
    if len(clean) < len(embeddings):
        skipped = set(embeddings) - set(clean)
        logger.warning(f"plot_pca_scatter: skipping non-finite models: {skipped}")
    if not clean:
        logger.warning("plot_pca_scatter: no valid embeddings, skipping")
        return

    rng = np.random.default_rng(MINIBATCH_SEED)
    names = list(clean.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        X = clean[name]
        # Subsample for scatter plot only — does not affect PCA eigenvalue analysis
        if X.shape[0] > max_scatter_samples:
            idx = rng.choice(X.shape[0], size=max_scatter_samples, replace=False)
            X_plot = X[idx]
        else:
            X_plot = X
        color = MODEL_COLORS.get(name, "#555")
        pca = skPCA(n_components=2)
        Z = pca.fit_transform(X_plot)
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.35, s=12, color=color, linewidths=0)
        var = pca.explained_variance_ratio_
        ax.set_title(name, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=8)

    n_shown = min(max_scatter_samples, min(v.shape[0] for v in clean.values()))
    fig.suptitle(
        f"2D PCA Projections  ·  {n_shown:,} utterances per model",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = plots_dir / "pca_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_cka_heatmap(cka_matrix: np.ndarray, names: list, plots_dir: Path):
    _apply_style()
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.0), max(6.5, len(names) * 0.9)))

    cmap = LinearSegmentedColormap.from_list(
        "cka", ["#FFFFFF", "#A5D6A7", "#2E7D32"], N=256
    )
    im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA  (0 = no similarity,  1 = identical)", fontsize=9)

    n = len(names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = cka_matrix[i, j]
            text_color = "black" if val < 0.7 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    # Draw separator between audio and text model groups
    audio_indices = [
        i for i, name in enumerate(names)
        if MODELS[name]["modality"] in AUDIO_MODALITIES
    ]
    if audio_indices:
        boundary = max(audio_indices) + 0.5
        ax.axhline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.axvline(boundary, color="#555", linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_title(
        "Pairwise Minibatch Linear CKA\n"
        "Dashed line separates audio/speech models from text-only LLMs",
        fontsize=12, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    path = plots_dir / "cka_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_cka_bar_cross_modal(cka_matrix: np.ndarray, names: list, plots_dir: Path):
    """
    Grouped bar chart: for each text LLM, show its CKA with each audio/speech model.
    """
    _apply_style()
    audio_names = [n for n in names if MODELS[n]["modality"] in AUDIO_MODALITIES]
    lm_names    = [n for n in names if MODELS[n]["modality"] not in AUDIO_MODALITIES]
    if not audio_names or not lm_names:
        return

    x = np.arange(len(lm_names))
    n_audio = len(audio_names)
    width = 0.7 / n_audio

    fig, ax = plt.subplots(figsize=(max(8, len(lm_names) * 1.8 + n_audio * 0.4), 5))

    for k, audio_name in enumerate(audio_names):
        a_idx = names.index(audio_name)
        scores = [cka_matrix[a_idx, names.index(lm)] for lm in lm_names]
        offsets = x + (k - (n_audio - 1) / 2) * width
        color = MODEL_COLORS.get(audio_name, "#555")
        bars = ax.bar(offsets, scores, width=width * 0.9, color=color,
                      label=audio_name, edgecolor="white", linewidth=0.8)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{score:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(lm_names, fontsize=10)
    max_val = max(
        cka_matrix[names.index(a), names.index(lm)]
        for a in audio_names for lm in lm_names
    )
    ax.set_ylim(0, min(1.0, max_val * 1.35 + 0.05))
    ax.set_ylabel("CKA with audio/speech model", fontsize=11)
    ax.set_title(
        "Cross-Modal CKA: Audio/Speech Models vs. Each Text LLM\n"
        "Higher = more similar representational geometry",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, framealpha=0.9, title="Audio/speech model")
    plt.tight_layout()
    path = plots_dir / "cka_cross_modal_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {path}")


def plot_stability(stability_results: dict, plots_dir: Path):
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
    bp = ax.boxplot(data, patch_artist=True, vert=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, std in zip(bp["boxes"], stds_sorted):
        intensity = min(std / 0.05, 1.0)
        patch.set_facecolor((0.2 + 0.6 * intensity, 0.6 - 0.4 * intensity,
                             0.8 - 0.6 * intensity, 0.5))
    ax.set_xticks(range(1, len(pairs) + 1))
    ax.set_xticklabels([p.replace(" vs ", "\nvs\n") for p in pairs], fontsize=7)
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
    path1 = data_dir / "cka_matrix.csv"
    with open(path1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, row_name in enumerate(names):
            w.writerow([row_name] + [f"{cka_matrix[i, j]:.6f}" for j in range(len(names))])
    logger.info(f"  Saved → {path1}")

    # Cross-modal table: all audio/speech models vs all text LLMs
    audio_names = [n for n in names if MODELS[n]["modality"] in AUDIO_MODALITIES]
    lm_names    = [n for n in names if MODELS[n]["modality"] not in AUDIO_MODALITIES]
    if audio_names and lm_names:
        path2 = data_dir / "cka_cross_modal.csv"
        with open(path2, "w", newline="") as f:
            w = csv.writer(f)
            header = ["lm_model", "params", "arch", "corpus"] + audio_names
            w.writerow(header)
            for lm in lm_names:
                lm_idx = names.index(lm)
                row = [lm, MODELS[lm]["params"], MODELS[lm]["arch"], MODELS[lm]["corpus"]]
                row += [f"{cka_matrix[names.index(a), lm_idx]:.6f}" for a in audio_names]
                w.writerow(row)
        logger.info(f"  Saved → {path2}")

    path3 = data_dir / "pca_summary.csv"
    with open(path3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "arch", "corpus",
                    "embedding_dim", "effective_rank",
                    "var_pc1", "var_top5", "var_top10"])
        for name, eigs in eigenvalues.items():
            dim = embeddings[name].shape[1] if name in embeddings else "—"
            w.writerow([
                name, MODELS[name]["params"], MODELS[name]["arch"], MODELS[name]["corpus"],
                dim, f"{effective_rank(eigs):.2f}",
                f"{eigs[0]:.4f}", f"{eigs[:5].sum():.4f}", f"{eigs[:10].sum():.4f}",
            ])
    logger.info(f"  Saved → {path3}")

    if stability_results:
        path4 = data_dir / "cka_stability.csv"
        with open(path4, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "cka_mean", "cka_std", "cka_min", "cka_max", "unstable"])
            for pair, stats in stability_results.items():
                scores = stats["scores"]
                w.writerow([
                    pair, f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
                    f"{min(scores):.6f}", f"{max(scores):.6f}",
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
    print("CROSS-MODAL CKA  (audio/speech ↔ text LLMs)")
    print(f"{'='*70}")
    audio_names = [n for n in names if MODELS[n]["modality"] in AUDIO_MODALITIES]
    lm_names    = [n for n in names if MODELS[n]["modality"] not in AUDIO_MODALITIES]
    for a in audio_names:
        a_idx = names.index(a)
        print(f"\n  {a}:")
        for lm in lm_names:
            score = cka_matrix[a_idx, names.index(lm)]
            meta = MODELS[lm]
            print(
                f"    ↔ {lm:<22} "
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
    print("  Cross-modal CKA > ~0.3 supports the Platonic Representation Hypothesis")
    print("  (convergence across modalities, Huh et al. 2024).")
    print("  Whisper-enc vs Whisper-dec CKA reveals enc/dec representational divergence.")
    print("  ⚠  std > 0.03 across subsets → possible outlier sensitivity")
    print("     (see Horoi et al. 'Deceiving the CKA Similarity Measure')")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-modal representation analysis with minibatch CKA"
    )
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS,
                   choices=list(SPLIT_CONFIG_MAP.keys()),
                   help="LibriSpeech splits (default: all)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap total samples (default: no cap)")
    p.add_argument("--root_dir", type=str, default=".",
                   help="Root dir; Data/ and Plots/ created here")
    p.add_argument("--skip_extraction", action="store_true",
                   help="Load cached embeddings instead of re-extracting")
    p.add_argument("--batch_size", type=int, default=MINIBATCH_SIZE,
                   help=f"CKA minibatch size (default: {MINIBATCH_SIZE})")
    p.add_argument("--lm_batch_size", type=int, default=32,
                   help="Batch size for text LM forward passes (default: 32)")
    p.add_argument("--whisper_batch_size", type=int, default=64,
                   help="Batch size for Whisper extraction (default: 64)")
    p.add_argument("--parakeet_batch_size", type=int, default=32,
                   help="Batch size for Parakeet extraction (default: 32)")
    p.add_argument("--mimi_batch_size", type=int, default=64,
                   help="Batch size for Mimi codec extraction (default: 64, reduce if OOM)")
    p.add_argument("--voxtral_batch_size", type=int, default=8,
                   help="Batch size for Voxtral extraction (default: 8, reduce if OOM)")
    p.add_argument("--higgs_batch_size", type=int, default=8,
                   help="Batch size for Higgs Audio V2 extraction (default: 8, reduce if OOM)")
    p.add_argument("--num_proc", type=int, default=8,
                   help="CPU workers for dataset shard loading (default: 8, set to ~24 on 28-core machines)")
    p.add_argument("--prefetch_queue_depth", type=int, default=8,
                   help="Background prefetch depth (default: 8; higher keeps GPU busier on fast CPUs)")
    p.add_argument("--pca_components", type=int, default=PCA_COMPONENTS,
                   help=f"PCA components for eigenvalue analysis (default: {PCA_COMPONENTS})")
    p.add_argument("--pca_plot_max_pc", type=int, default=PCA_PLOT_MAX_PC,
                   help=f"Max PC shown on x-axis of eigenspectra plots (default: {PCA_PLOT_MAX_PC})")
    p.add_argument("--run_stability", action="store_true",
                   help="Run outlier stability check (off by default — adds significant runtime)")
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
        dataset, texts = load_librispeech(
            args.splits, args.max_samples,
            cache_dir=data_dir, num_proc=args.num_proc,
        )
    N = len(texts)

    # ------------------------------------------------------------------
    # 2. Extract / load embeddings
    # ------------------------------------------------------------------
    embeddings = {}
    for model_name, cfg in MODELS.items():
        cache_path = data_dir / f"embeddings_{model_name}.pkl"

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
        logger.info(f"Extracting: {model_name}  [{cfg['params']} | {cfg['arch']} | {cfg['corpus']}]")
        logger.info("=" * 60)

        with timer(f"Extract {model_name}"):
            try:
                modality = cfg["modality"]
                if modality == "audio-whisper-enc":
                    emb = extract_whisper_encoder_embeddings(
                        cfg["hf_id"], dataset, device,
                        batch_size=args.whisper_batch_size,
                        checkpoint_dir=data_dir,
                        prefetch_queue_depth=args.prefetch_queue_depth,
                    )
                elif modality == "audio-whisper-dec":
                    emb = extract_whisper_decoder_embeddings(
                        cfg["hf_id"], dataset, device,
                        batch_size=args.whisper_batch_size,
                        checkpoint_dir=data_dir,
                        prefetch_queue_depth=args.prefetch_queue_depth,
                    )
                elif modality == "audio-parakeet":
                    emb = extract_parakeet_embeddings(
                        cfg["hf_id"], dataset, device,
                        batch_size=args.parakeet_batch_size,
                        checkpoint_dir=data_dir,
                        prefetch_queue_depth=args.prefetch_queue_depth,
                    )
                elif modality == "audio-mimi":
                    emb = extract_mimi_embeddings(
                        cfg["hf_id"], dataset, device,
                        batch_size=args.mimi_batch_size,
                        checkpoint_dir=data_dir,
                        prefetch_queue_depth=args.prefetch_queue_depth,
                    )
                elif modality == "tts-higgs":
                    emb = extract_higgs_audio_embeddings(
                        cfg["hf_id"], texts, device,
                        batch_size=args.higgs_batch_size,
                        checkpoint_dir=data_dir,
                        prefetch_queue_depth=args.prefetch_queue_depth,
                    )
                elif modality == "tts-qwen3":
                    emb = extract_qwen3_tts_embeddings(
                        cfg["hf_id"], texts, device,
                        batch_size=args.lm_batch_size,
                        checkpoint_dir=data_dir,
                        prefetch_queue_depth=args.prefetch_queue_depth,
                    )
                elif modality == "audio-voxtral":
                    emb = extract_voxtral_embeddings(
                        model_name, cfg["hf_id"], dataset, device,
                        batch_size=args.voxtral_batch_size,
                        checkpoint_dir=data_dir,
                        prefetch_queue_depth=args.prefetch_queue_depth,
                    )
                else:  # text
                    emb = extract_lm_embeddings(
                        cfg["hf_id"], texts, device,
                        batch_size=args.lm_batch_size,
                        checkpoint_dir=data_dir,
                        prefetch_queue_depth=args.prefetch_queue_depth,
                    )
            except Exception as e:
                logger.error(f"Extraction failed for {model_name}: {e}. Skipping.")
                release_vram(f"after {model_name} failure")
                continue

        embeddings[model_name] = emb

        n_bad = (~np.isfinite(emb)).sum()
        if n_bad > 0:
            logger.error(f"{model_name}: {n_bad} non-finite values — NOT caching.")
            del embeddings[model_name]
            continue

        with open(cache_path, "wb") as f:
            pickle.dump(emb, f)
        logger.info(f"Cached → {cache_path}  ({cache_path.stat().st_size / 1024**2:.1f} MB)")

    names = list(embeddings.keys())
    logger.info(f"Models with embeddings: {names}")

    total_bytes = sum(v.nbytes for v in embeddings.values())
    logger.info(
        f"Total embedding RAM: {total_bytes / 1024**3:.1f} GB  "
        f"({len(names)} models × {N:,} samples)"
    )

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
            logger.info(f"  {name:<22}  dim={X.shape[1]:>5}  n={X.shape[0]:>7,}  eff_rank={er:.1f}")

    with timer("PCA plots"):
        plot_eigenspectra(eigenvalues, plots_dir, max_pc=args.pca_plot_max_pc)
        plot_eigenspectra_overlay(eigenvalues, plots_dir, max_pc=args.pca_plot_max_pc)
        plot_effective_rank_bar(eigenvalues, plots_dir)
        try:
            from sklearn.decomposition import PCA  # noqa: F401
            plot_pca_scatter(embeddings, plots_dir)
        except ImportError:
            logger.warning("scikit-learn not installed — skipping PCA scatter")

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
                t_pair = time.perf_counter()
                score = minibatch_cka(embeddings[names[i]], embeddings[names[j]],
                                      batch_size=args.batch_size)
                cka_matrix[i, j] = score
                key = f"{names[i]} vs {names[j]}"
                cka_results[key] = score
                logger.info(f"  CKA({names[i]}, {names[j]}) = {score:.4f}  ({time.perf_counter()-t_pair:.1f}s)")
                pair_bar.update(1)
        pair_bar.close()

    with timer("CKA plots"):
        plot_cka_heatmap(cka_matrix, names, plots_dir)
        plot_cka_bar_cross_modal(cka_matrix, names, plots_dir)

    # ------------------------------------------------------------------
    # 5. Outlier stability check
    # ------------------------------------------------------------------
    stability_results = {}
    if args.run_stability:
        logger.info("=" * 60)
        logger.info(f"Stability check  ({N_STABILITY_SUBSETS} subsets × {int(STABILITY_SUBSET_FRAC*100)}%)")
        logger.info("=" * 60)
        upper_pairs = [(names[i], names[j]) for i in range(n) for j in range(n) if j > i]
        n_stability_jobs = len(upper_pairs) * N_STABILITY_SUBSETS
        with timer("Stability"):
            stab_bar = tqdm(total=n_stability_jobs, desc="Stability subsets", unit="subset")
            for name_a, name_b in upper_pairs:
                pair_key = f"{name_a} vs {name_b}"
                stats = cka_stability_check(embeddings[name_a], embeddings[name_b],
                                            batch_size=args.batch_size)
                stability_results[pair_key] = stats
                flag = "  ⚠ HIGH VARIANCE" if stats["std"] > 0.03 else ""
                logger.info(f"  {pair_key:<42}  mean={stats['mean']:.4f}  std={stats['std']:.4f}{flag}")
                stab_bar.update(N_STABILITY_SUBSETS)
            stab_bar.close()
        with timer("Stability plot"):
            plot_stability(stability_results, plots_dir)
    else:
        logger.info("Stability check skipped (pass --run_stability to enable)")

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Saving summary tables")
    logger.info("=" * 60)
    with timer("Save tables"):
        save_summary_tables(cka_matrix, names, eigenvalues, embeddings,
                            stability_results, data_dir)

    results_json = {
        "n_samples": N,
        "splits_used": args.splits,
        "minibatch_size": args.batch_size,
        "pca_plot_max_pc": args.pca_plot_max_pc,
        "models": {k: v for k, v in MODELS.items()},
        "embedding_shapes": {k: list(v.shape) for k, v in embeddings.items()},
        "cka_scores": cka_results,
        "cka_matrix": cka_matrix.tolist(),
        "eigenvalue_spectra": {k: v.tolist() for k, v in eigenvalues.items()},
        "effective_ranks": {k: effective_rank(v) for k, v in eigenvalues.items()},
        "stability": {k: {"mean": v["mean"], "std": v["std"]}
                      for k, v in stability_results.items()},
    }
    json_path = data_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results JSON → {json_path}")

    print_summary(cka_matrix, names, stability_results)

    total_elapsed = time.perf_counter() - run_start
    m, s = divmod(total_elapsed, 60)
    h, m = divmod(m, 60)
    logger.info(f"Run complete — total time: {int(h)}h {int(m)}m {s:.1f}s")
    logger.info(f"Data/  → {data_dir.resolve()}")
    logger.info(f"Plots/ → {plots_dir.resolve()}")
    logger.info(f"Log    → {log_path}")


if __name__ == "__main__":
    main()
