#!/usr/bin/env python3
"""
Whisper Layer-wise Cosine Similarity Analysis

For each Whisper model (base / small / medium / large-v1/v2/v3, encoder and
decoder), extracts word-level embeddings from every hidden layer and plots
per-layer cosine similarity histograms using the same cross-speaker,
same-word/sentence pairing logic as generate_similarity_histograms.py.

Requires a previous word-level analysis run so that word_records and audio
WAVs already exist on disk.  Does NOT re-run alignment or audio download.

All per-layer embeddings are cached as PKL files so you can re-run plotting
without re-extracting.

Usage
-----
  # MCV dataset (cross-speaker, recommended)
  python whisper_layer_similarity.py \\
      --root_dir /opt/modeling/zhoughton/misc/UGE-audio-text-models \\
      --dataset mcv

  # LJSpeech or MLS
  python whisper_layer_similarity.py --root_dir /path/to/root --dataset word
  python whisper_layer_similarity.py --root_dir /path/to/root --dataset mls

  # Skip extraction if PKL files already exist
  python whisper_layer_similarity.py --root_dir ... --dataset mcv --skip_extraction

  # Only plot (no extraction, no similarity re-computation)
  python whisper_layer_similarity.py --root_dir ... --dataset mcv --skip_extraction
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHISPER_SR      = 16000
WHISPER_ENC_FPS = 50.0
MAX_TEXT_TOKENS = 256

WHISPER_MODELS = {
    "whisper-base":       "openai/whisper-base",
    "whisper-small":      "openai/whisper-small",
    "whisper-medium":     "openai/whisper-medium",
    "whisper-large-v1":   "openai/whisper-large",
    "whisper-large-v2":   "openai/whisper-large-v2",
    "whisper-large-v3":   "openai/whisper-large-v3",
}

# Dataset configs: data_dir name, plots_dir name, file prefix, audio subdir,
# word_records filename, utterance loader tag
DATASET_CONFIGS = {
    "mcv": {
        "data_dir":     "MCVData",
        "plots_dir":    "MCVPlots",
        "prefix":       "mcv_",
        "audio_subdir": "mcv_sample",
        "word_records": "mcv_word_records.json",
        "loader":       "mcv",
    },
    "mls": {
        "data_dir":     "MLSData",
        "plots_dir":    "MLSPlots",
        "prefix":       "mls_",
        "audio_subdir": "mls_sample",
        "word_records": "mls_word_records.json",
        "loader":       "mls",
    },
    "word": {
        "data_dir":     "WordData",
        "plots_dir":    "WordPlots",
        "prefix":       "word_",
        "audio_subdir": "LJSpeech-1.1",
        "word_records": "word_records.json",
        "loader":       "ljspeech",
    },
}

logger = logging.getLogger("whisper_layer_sim")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(ch)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        logger.info("No GPU found — using CPU")
    return dev


# ---------------------------------------------------------------------------
# Audio loaders
# ---------------------------------------------------------------------------

def _load_wav_dir(wavs_dir: Path, transcripts: dict, metainfo: dict | None) -> dict:
    utterances = {}
    for utt_id, text in tqdm(transcripts.items(), desc="Loading audio", unit="utt"):
        wav_path = wavs_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            continue
        waveform, sr = torchaudio.load(str(wav_path))
        audio = waveform.squeeze(0).numpy()
        entry = {"audio": audio, "sr": int(sr), "text": text}
        if metainfo and utt_id in metainfo:
            entry["speaker_id"] = metainfo[utt_id].get("speaker_id", "unknown")
        utterances[utt_id] = entry
    return utterances


def load_utterances(loader: str, data_dir: Path, audio_subdir: str) -> dict:
    sample_dir = data_dir / audio_subdir
    transcripts_path = sample_dir / "transcripts.json"

    if loader in ("mcv", "mls"):
        if not transcripts_path.exists():
            raise FileNotFoundError(f"transcripts.json not found at {transcripts_path}")
        with open(transcripts_path) as f:
            transcripts = json.load(f)
        metainfo = None
        metainfo_path = sample_dir / "metainfo.json"
        if metainfo_path.exists():
            with open(metainfo_path) as f:
                metainfo = json.load(f)
        wavs_dir = sample_dir / "wavs"
        utterances = _load_wav_dir(wavs_dir, transcripts, metainfo)

    elif loader == "ljspeech":
        metadata_path = sample_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
        transcripts = {}
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) >= 2:
                    utt_id = parts[0]
                    text   = parts[2] if len(parts) >= 3 and parts[2] else parts[1]
                    transcripts[utt_id] = text.lower()
        wavs_dir = sample_dir / "wavs"
        utterances = _load_wav_dir(wavs_dir, transcripts, None)

    else:
        raise ValueError(f"Unknown loader: {loader}")

    logger.info(f"Loaded {len(utterances):,} utterances")
    return utterances


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    t = torch.from_numpy(audio).unsqueeze(0)
    t = torchaudio.functional.resample(t, orig_freq=src_sr, new_freq=dst_sr)
    return t.squeeze(0).numpy()


def _ensure_resampled(utterances: dict, utt_ids: list, target_sr: int) -> str:
    key = f"audio_{target_sr}"
    for uid in utt_ids:
        if key not in utterances[uid]:
            utt = utterances[uid]
            utterances[uid][key] = _resample(utt["audio"], utt["sr"], target_sr)
    return key


def _time_to_frame(t: float, fps: float, max_frame: int) -> int:
    return min(int(t * fps), max_frame)


def _slice_frames(hidden: np.ndarray, start: float, end: float, fps: float) -> np.ndarray | None:
    T  = hidden.shape[0]
    f0 = _time_to_frame(start, fps, T)
    f1 = max(f0 + 1, _time_to_frame(end, fps, T))
    f1 = min(f1, T)
    if f0 >= T:
        return None
    return hidden[f0:f1].mean(axis=0)


# ---------------------------------------------------------------------------
# Layer embedding extraction
# ---------------------------------------------------------------------------

def _group_by_utt(word_records):
    groups = {}
    for idx, rec in enumerate(word_records):
        groups.setdefault(rec["utt_id"], []).append(idx)
    return groups


def extract_encoder_layers(
    model_name: str, model_id: str, word_records: list,
    utterances: dict, device: torch.device,
    batch_size: int = 64,
) -> dict[int, np.ndarray]:
    """Returns {layer_idx: embeddings_array (N, D)} for all encoder layers."""
    logger.info(f"Extracting encoder layers: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model     = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()

    n_layers = model.config.encoder_layers
    logger.info(f"  Encoder layers: {n_layers}")

    utt_to_words = _group_by_utt(word_records)
    utt_ids      = list(utt_to_words.keys())
    audio_key    = _ensure_resampled(utterances, utt_ids, WHISPER_SR)
    N            = len(word_records)

    # layer_idx -> list of (word_idx, emb) pairs
    layer_accum: dict[int, list] = {i: [] for i in range(n_layers + 1)}
    completed_words: set = set()

    for batch_start in tqdm(range(0, len(utt_ids), batch_size),
                            desc=f"{model_name} enc", unit="batch"):
        batch_ids    = utt_ids[batch_start : batch_start + batch_size]
        audio_arrays = [utterances[uid][audio_key] for uid in batch_ids]

        inputs   = processor(audio_arrays, sampling_rate=WHISPER_SR, return_tensors="pt")
        features = inputs["input_features"].to(device, dtype=torch.float16)

        with torch.no_grad():
            out = model.encoder(features, output_hidden_states=True)

        # hidden_states: tuple of (batch, seq_len, dim), length = n_layers + 1
        # Index 0 = post-conv projection; 1..n_layers = transformer layer outputs
        hidden_all = [h.float().cpu().numpy() for h in out.hidden_states]

        for b, utt_id in enumerate(batch_ids):
            for word_idx in utt_to_words[utt_id]:
                if word_idx in completed_words:
                    continue
                rec = word_records[word_idx]
                for layer_idx, hidden in enumerate(hidden_all):
                    emb = _slice_frames(hidden[b], rec["start"], rec["end"], WHISPER_ENC_FPS)
                    if emb is not None:
                        layer_accum[layer_idx].append((word_idx, emb))
                completed_words.add(word_idx)

    del model
    torch.cuda.empty_cache()

    results = {}
    for layer_idx, pairs in layer_accum.items():
        if not pairs:
            continue
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        D = pairs_sorted[0][1].shape[0]
        arr = np.zeros((N, D), dtype=np.float32)
        for idx, emb in pairs_sorted:
            arr[idx] = emb
        results[layer_idx] = arr

    return results


def extract_decoder_layers(
    model_name: str, model_id: str, word_records: list,
    utterances: dict, device: torch.device,
    batch_size: int = 32,
) -> dict[int, np.ndarray]:
    """Returns {layer_idx: embeddings_array (N, D)} for all decoder layers."""
    logger.info(f"Extracting decoder layers: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    model     = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.decoder_layers
    logger.info(f"  Decoder layers: {n_layers}")

    utt_to_words = _group_by_utt(word_records)
    utt_ids      = list(utt_to_words.keys())
    audio_key    = _ensure_resampled(utterances, utt_ids, WHISPER_SR)
    N            = len(word_records)

    layer_accum: dict[int, list] = {i: [] for i in range(n_layers + 1)}
    completed_words: set = set()

    for batch_start in tqdm(range(0, len(utt_ids), batch_size),
                            desc=f"{model_name} dec", unit="batch"):
        batch_ids = utt_ids[batch_start : batch_start + batch_size]
        sentences = [word_records[utt_to_words[uid][0]]["sentence"] for uid in batch_ids]

        try:
            audio_arrays = [utterances[uid][audio_key] for uid in batch_ids]
            audio_inputs = processor(audio_arrays, sampling_rate=WHISPER_SR, return_tensors="pt")
            features     = audio_inputs["input_features"].to(device, dtype=torch.float16)

            with torch.no_grad():
                encoder_hidden = model.encoder(features).last_hidden_state

            text_enc = tokenizer(sentences, return_tensors="pt", truncation=True,
                                 max_length=MAX_TEXT_TOKENS, padding=True,
                                 return_offsets_mapping=True)
            offset_mapping    = text_enc.pop("offset_mapping").tolist()
            decoder_input_ids = text_enc["input_ids"].to(device)
            decoder_attn_mask = text_enc["attention_mask"].to(device)

            with torch.no_grad():
                dec_out = model.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attn_mask,
                    encoder_hidden_states=encoder_hidden,
                    output_hidden_states=True,
                )

            hidden_all = [h.float().cpu().numpy() for h in dec_out.hidden_states]

            for b, utt_id in enumerate(batch_ids):
                om = offset_mapping[b]
                for word_idx in utt_to_words[utt_id]:
                    if word_idx in completed_words:
                        continue
                    rec    = word_records[word_idx]
                    char_s = rec["char_start"]
                    char_e = rec["char_end"]
                    if char_s < 0:
                        continue
                    token_indices = [t for t, (ts, te) in enumerate(om)
                                     if ts < char_e and te > char_s and ts < te]
                    if not token_indices:
                        continue
                    for layer_idx, hidden in enumerate(hidden_all):
                        emb = hidden[b][token_indices].mean(axis=0)
                        layer_accum[layer_idx].append((word_idx, emb))
                    completed_words.add(word_idx)

        except Exception as e:
            logger.warning(f"Skipping batch at {batch_start}: {e}")

    del model
    torch.cuda.empty_cache()

    results = {}
    for layer_idx, pairs in layer_accum.items():
        if not pairs:
            continue
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        D = pairs_sorted[0][1].shape[0]
        arr = np.zeros((N, D), dtype=np.float32)
        for idx, emb in pairs_sorted:
            arr[idx] = emb
        results[layer_idx] = arr

    return results


# ---------------------------------------------------------------------------
# Cosine similarity (per-group, cross-speaker)
# ---------------------------------------------------------------------------

def per_group_cosine_sims(
    emb: np.ndarray,
    word_ids: np.ndarray,
    speaker_ids: np.ndarray | None,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return all cross-speaker cosine similarity values for same-word groups."""
    X = emb[valid_mask].astype(np.float32)
    w = word_ids[valid_mask]
    s = speaker_ids[valid_mask] if speaker_ids is not None else None

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-10)

    all_vals = []
    for wid in np.unique(w):
        idx = np.where(w == wid)[0]
        if len(idx) < 2:
            continue
        Xg = X[idx]
        sim = Xg @ Xg.T
        n_g = len(idx)
        if s is not None:
            sg = s[idx]
            cross = sg[:, None] != sg[None, :]
        else:
            cross = np.ones((n_g, n_g), dtype=bool)
        mask = cross & ~np.eye(n_g, dtype=bool)
        if mask.any():
            all_vals.append(sim[mask])

    return np.concatenate(all_vals) if all_vals else np.array([])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_layer_histograms(
    model_name: str,
    layer_sims: dict[int, np.ndarray],
    out_dir: Path,
    prefix: str,
):
    """One histogram per layer, plus a mean-per-layer line plot."""
    _apply_style()
    n_layers = len(layer_sims)
    if n_layers == 0:
        return

    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    layer_indices = sorted(layer_sims.keys())

    # Individual histogram per layer
    for layer_idx in layer_indices:
        vals = layer_sims[layer_idx]
        if len(vals) == 0:
            continue
        mean_val = float(vals.mean())
        clipped  = np.clip(vals, -1, 1)
        _, ax = plt.subplots(figsize=(7, 4))
        ax.hist(clipped, bins=100, range=(-1, 1),
                color=colors[layer_indices.index(layer_idx)], edgecolor="none", alpha=0.85)
        ax.axvline(mean_val, color="#E53935", linewidth=1.5, linestyle="--",
                   label=f"mean = {mean_val:.3f}")
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Cosine Similarity", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{model_name}  —  layer {layer_idx}  (n={len(vals):,})", fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = out_dir / f"{prefix}{model_name}_layer{layer_idx:02d}_cosine_hist.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    # Faceted grid: all layers for this model
    ncols = min(6, n_layers)
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.2), squeeze=False)
    axes_flat = axes.flatten()
    for i, layer_idx in enumerate(layer_indices):
        vals = layer_sims[layer_idx]
        ax   = axes_flat[i]
        if len(vals) > 0:
            ax.hist(np.clip(vals, -1, 1), bins=60, range=(-1, 1),
                    color=colors[i], edgecolor="none", alpha=0.85)
        ax.set_xlim(-1, 1)
        ax.set_title(f"Layer {layer_idx}", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
    for i in range(n_layers, len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle(f"{model_name}  —  cosine similarity per layer", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"{prefix}{model_name}_all_layers_cosine_faceted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved faceted plot → {path}")

    # Mean cosine similarity vs layer depth
    means = [float(layer_sims[li].mean()) if len(layer_sims[li]) > 0 else float("nan")
             for li in layer_indices]
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layer_indices, means, marker="o", linewidth=2, markersize=5, color="#0277BD")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=11)
    ax.set_title(f"{model_name}  —  mean cosine similarity vs layer depth", fontsize=11)
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_indices, fontsize=7)
    plt.tight_layout()
    path = out_dir / f"{prefix}{model_name}_mean_cosine_by_layer.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved mean-by-layer plot → {path}")


def plot_all_models_mean_by_layer(
    all_means: dict[str, tuple[list, list]],
    out_dir: Path,
    prefix: str,
):
    """Overlay line plot: mean cosine similarity vs layer for all models."""
    _apply_style()
    if not all_means:
        return

    n = len(all_means)
    colors = plt.cm.tab20(np.linspace(0, 1, n))
    _, ax = plt.subplots(figsize=(12, 5))
    for (model_name, (layer_indices, means)), color in zip(all_means.items(), colors):
        ax.plot(layer_indices, means, marker="o", linewidth=1.5, markersize=4,
                label=model_name, color=color)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=11)
    ax.set_title("Mean cross-speaker cosine similarity vs layer depth — all models",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    path = out_dir / f"{prefix}all_models_mean_cosine_by_layer.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved overlay plot → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Whisper layer-wise cosine similarity analysis")
    p.add_argument("--root_dir", default=".", type=Path,
                   help="Root directory (same as used for the word-level analyses)")
    p.add_argument("--dataset", default="mcv", choices=list(DATASET_CONFIGS),
                   help="Which dataset to use: mcv | mls | word  (default: mcv)")
    p.add_argument("--models", nargs="+", default=list(WHISPER_MODELS),
                   help="Which Whisper models to process (default: all)")
    p.add_argument("--enc_batch_size", default=64, type=int)
    p.add_argument("--dec_batch_size", default=32, type=int)
    p.add_argument("--skip_extraction", action="store_true",
                   help="Skip embedding extraction; use cached PKL files only")
    return p.parse_args()


def main():
    setup_logging()
    args   = parse_args()
    root   = Path(args.root_dir)
    cfg    = DATASET_CONFIGS[args.dataset]

    data_dir  = root / cfg["data_dir"]
    plots_dir = root / cfg["plots_dir"]
    layer_dir = plots_dir / "layer_similarity"
    layer_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg["prefix"]

    # ── Load word records ────────────────────────────────────────────────────
    wr_path = data_dir / cfg["word_records"]
    if not wr_path.exists():
        logger.error(f"Word records not found: {wr_path}")
        logger.error("Run the word-level analysis script first.")
        sys.exit(1)
    with open(wr_path) as f:
        word_records = json.load(f)
    logger.info(f"Loaded {len(word_records):,} word records from {wr_path}")

    # ── Build word_ids and speaker_ids ────────────────────────────────────────
    all_types: dict = {}
    word_ids_list = []
    for rec in word_records:
        key = (rec["word"], rec["sentence"], rec["char_start"])
        if key not in all_types:
            all_types[key] = len(all_types)
        word_ids_list.append(all_types[key])
    word_ids = np.array(word_ids_list, dtype=np.int32)

    speaker_ids = None
    if all("speaker_id" in rec for rec in word_records[:10]):
        all_speakers = sorted(set(rec.get("speaker_id", "unknown") for rec in word_records))
        spk_to_id    = {s: i for i, s in enumerate(all_speakers)}
        speaker_ids  = np.array(
            [spk_to_id[rec.get("speaker_id", "unknown")] for rec in word_records],
            dtype=np.int32
        )
        logger.info(f"Speakers: {len(all_speakers):,}")

    # ── Load audio ───────────────────────────────────────────────────────────
    if not args.skip_extraction:
        utterances = load_utterances(cfg["loader"], data_dir, cfg["audio_subdir"])
    else:
        utterances = {}

    device = get_device()

    # ── Per-model extraction + plotting ──────────────────────────────────────
    all_means: dict[str, tuple[list, list]] = {}

    for short_name in args.models:
        if short_name not in WHISPER_MODELS:
            logger.warning(f"Unknown model '{short_name}', skipping")
            continue
        model_id = WHISPER_MODELS[short_name]

        for component in ("enc", "dec"):
            full_name = f"{short_name}-{component}"

            # ── Load or extract per-layer PKL files ──────────────────────────
            layer_cache_dir = data_dir / "layer_embeddings"
            layer_cache_dir.mkdir(exist_ok=True)

            # Check which layers are already cached
            def cache_path(layer_idx):
                return layer_cache_dir / f"{prefix}layer_emb_{full_name}_layer{layer_idx:02d}.pkl"

            layer_embs: dict[int, np.ndarray] = {}

            # Try to load any existing caches first
            for existing in sorted(layer_cache_dir.glob(
                    f"{prefix}layer_emb_{full_name}_layer*.pkl")):
                try:
                    li = int(existing.stem.split("layer")[-1])
                    with open(existing, "rb") as f:
                        layer_embs[li] = pickle.load(f)
                except Exception:
                    pass

            if layer_embs:
                logger.info(f"  {full_name}: loaded {len(layer_embs)} cached layers")

            # Extract missing layers (unless skip_extraction)
            if not args.skip_extraction:
                if component == "enc":
                    extracted = extract_encoder_layers(
                        full_name, model_id, word_records, utterances, device,
                        batch_size=args.enc_batch_size,
                    )
                else:
                    extracted = extract_decoder_layers(
                        full_name, model_id, word_records, utterances, device,
                        batch_size=args.dec_batch_size,
                    )
                for li, arr in extracted.items():
                    if li not in layer_embs:
                        with open(cache_path(li), "wb") as f:
                            pickle.dump(arr, f)
                        layer_embs[li] = arr
                logger.info(f"  {full_name}: {len(extracted)} layers extracted and cached")

            if not layer_embs:
                logger.warning(f"  {full_name}: no embeddings available, skipping")
                continue

            # ── Compute cosine similarities ───────────────────────────────────
            logger.info(f"  Computing cosine similarities: {full_name}")
            valid_mask = np.ones(len(word_records), dtype=bool)
            for arr in layer_embs.values():
                valid_mask &= np.linalg.norm(arr, axis=1) > 1e-10

            layer_sims: dict[int, np.ndarray] = {}
            for li in sorted(layer_embs.keys()):
                vals = per_group_cosine_sims(
                    layer_embs[li], word_ids, speaker_ids, valid_mask
                )
                layer_sims[li] = vals

            # ── Plot ──────────────────────────────────────────────────────────
            plot_layer_histograms(full_name, layer_sims, layer_dir, prefix)

            layer_indices = sorted(layer_sims.keys())
            means = [float(layer_sims[li].mean()) if len(layer_sims[li]) > 0 else float("nan")
                     for li in layer_indices]
            all_means[full_name] = (layer_indices, means)

    # ── Overlay plot across all models ────────────────────────────────────────
    plot_all_models_mean_by_layer(all_means, layer_dir, prefix)
    logger.info("Done.")


if __name__ == "__main__":
    main()
