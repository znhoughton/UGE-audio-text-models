#!/usr/bin/env python3
"""
Whisper "Deaf" Decoder Analysis

Compares three families of word-level representations via CKA:
  1. Whisper encoders          — audio-conditioned
  2. Whisper decoders (normal) — audio + text (cross-attention to encoder)
  3. Whisper decoders (deaf)   — text only (encoder hidden states zeroed out)
  4. Other audio models        — Parakeet, Mimi
  5. Text LLMs                 — text only

The "deaf" decoder zeroes the encoder hidden states fed into the decoder
cross-attention.  The decoder then receives no audio signal and must rely
entirely on self-attention over text tokens, making it behave like an LLM.

The comparison answers: how much of the decoder's word representations are
audio-driven vs. text-driven?  If deaf decoders cluster with LLMs, the
decoder is fundamentally text-like.  If they cluster with normal decoders,
audio conditioning is not dominant.

Reuses cached embedding PKLs from prior word-level analysis runs.
Extracts and caches deaf decoder embeddings on first run.

Datasets
--------
  word  → WordData/  (LJSpeech, single speaker, random minibatch CKA)
  mcv   → MCVData/   (MCV cross-speaker, per-group CKA with word/speaker mask)

Usage
-----
  python whisper_deaf_decoder_analysis.py \\
      --root_dir /opt/modeling/zhoughton/misc/UGE-audio-text-models \\
      --dataset mcv

  python whisper_deaf_decoder_analysis.py --root_dir ... --dataset word

  # Skip extraction if deaf-decoder PKL files are already cached
  python whisper_deaf_decoder_analysis.py --root_dir ... --skip_extraction
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHISPER_SR      = 16000
MAX_TEXT_TOKENS = 256
MINIBATCH_SIZE  = 2048
MINIBATCH_SEED  = 42

# HuggingFace IDs for Whisper decoder models (keyed by the base model name
# used in the existing embedding PKL files, e.g. "whisper-large-dec")
WHISPER_DEC_HF_IDS = {
    "whisper-base-dec":       "openai/whisper-base",
    "whisper-small-dec":      "openai/whisper-small",
    "whisper-medium-dec":     "openai/whisper-medium",
    "whisper-large-dec":      "openai/whisper-large-v3",
    "whisper-large-v2-dec":   "openai/whisper-large-v2",
    "whisper-large-v1-dec":   "openai/whisper-large",
}

DATASET_CONFIGS = {
    "mcv": {
        "data_dir":       "MCVData",
        "plots_dir":      "MCVPlots",
        "prefix":         "mcv_",
        "pkl_prefix":     "mcv_",     # MCVData/mcv_word_embeddings_*.pkl
        "audio_subdir":   "mcv_sample",
        "word_records":   "mcv_word_records.json",
        "loader":         "mcv",
        "cross_speaker":  True,
    },
    "word": {
        "data_dir":       "WordData",
        "plots_dir":      "WordPlots",
        "prefix":         "word_",
        "pkl_prefix":     "",         # WordData/word_embeddings_*.pkl (no double prefix)
        "audio_subdir":   "LJSpeech-1.1",
        "word_records":   "word_records.json",
        "loader":         "ljspeech",
        "cross_speaker":  False,
    },
}

# Visual grouping for the heatmap annotation bar
def _model_group(name: str) -> str:
    if name.endswith("-deaf"):
        return "Deaf decoder"
    if name.endswith("-enc"):
        return "Encoder"
    if name.endswith("-dec"):
        return "Decoder"
    if name in ("parakeet-ctc-0.6b", "mimi"):
        return "Audio (other)"
    return "LLM"

GROUP_COLORS = {
    "Encoder":      "#0277BD",
    "Decoder":      "#00838F",
    "Deaf decoder": "#E65100",
    "Audio (other)": "#880E4F",
    "LLM":          "#4A148C",
}

logger = logging.getLogger("whisper_deaf")


# ---------------------------------------------------------------------------
# Logging / device
# ---------------------------------------------------------------------------

def setup_logging():
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(ch)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        logger.info("No GPU — using CPU")
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
        entry = {"audio": waveform.squeeze(0).numpy(), "sr": int(sr), "text": text}
        if metainfo and utt_id in metainfo:
            entry["speaker_id"] = metainfo[utt_id].get("speaker_id", "unknown")
        utterances[utt_id] = entry
    return utterances


def load_utterances(loader: str, data_dir: Path, audio_subdir: str) -> dict:
    sample_dir = data_dir / audio_subdir
    if loader in ("mcv", "mls"):
        with open(sample_dir / "transcripts.json") as f:
            transcripts = json.load(f)
        metainfo = None
        mp = sample_dir / "metainfo.json"
        if mp.exists():
            with open(mp) as f:
                metainfo = json.load(f)
        utterances = _load_wav_dir(sample_dir / "wavs", transcripts, metainfo)
    else:  # ljspeech
        metadata_path = sample_dir / "metadata.csv"
        transcripts = {}
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) >= 2:
                    utt_id = parts[0]
                    text   = parts[2] if len(parts) >= 3 and parts[2] else parts[1]
                    transcripts[utt_id] = text.lower()
        utterances = _load_wav_dir(sample_dir / "wavs", transcripts, None)
    logger.info(f"Loaded {len(utterances):,} utterances")
    return utterances


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


# ---------------------------------------------------------------------------
# Deaf decoder extraction
# ---------------------------------------------------------------------------

def _group_by_utt(word_records: list) -> dict:
    groups = {}
    for idx, rec in enumerate(word_records):
        groups.setdefault(rec["utt_id"], []).append(idx)
    return groups


def extract_deaf_decoder_embeddings(
    model_name: str,
    model_id: str,
    word_records: list,
    utterances: dict,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract word embeddings from a Whisper decoder with cross-attention zeroed.

    Registers a forward hook on every decoder cross-attention layer that replaces
    the attention output with exactly zero before the residual add.  The decoder
    then receives no audio signal whatsoever and operates purely via self-attention
    over the text token sequence.
    """
    logger.info(f"Loading {model_id} (deaf decoder)")
    processor = WhisperProcessor.from_pretrained(model_id)
    model     = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Hook every decoder cross-attention layer to zero its output exactly
    hooks = []
    for layer in model.decoder.layers:
        hooks.append(layer.encoder_attn.register_forward_hook(
            lambda m, inp, out: (torch.zeros_like(out[0]),) + out[1:]
        ))

    # Dummy encoder hidden states — shape is required by the decoder API but
    # content is irrelevant since the cross-attention output is zeroed by hooks
    d_model     = model.config.d_model
    enc_seq_len = model.config.max_source_positions  # 1500 for all Whisper sizes

    utt_to_words = _group_by_utt(word_records)
    utt_ids      = list(utt_to_words.keys())
    N            = len(word_records)
    _ensure_resampled(utterances, utt_ids, WHISPER_SR)

    word_embeddings_list = []
    completed_words: set = set()
    errors = 0

    for batch_start in tqdm(range(0, len(utt_ids), batch_size),
                            desc=model_name, unit="batch"):
        batch_ids = utt_ids[batch_start : batch_start + batch_size]
        B         = len(batch_ids)
        sentences = [word_records[utt_to_words[uid][0]]["sentence"] for uid in batch_ids]

        try:
            text_enc = tokenizer(
                sentences, return_tensors="pt", truncation=True,
                max_length=MAX_TEXT_TOKENS, padding=True,
                return_offsets_mapping=True,
            )
            offset_mapping    = text_enc.pop("offset_mapping").tolist()
            decoder_input_ids = text_enc["input_ids"].to(device)
            decoder_attn_mask = text_enc["attention_mask"].to(device)

            encoder_dummy = torch.zeros(
                B, enc_seq_len, d_model, device=device, dtype=torch.float16
            )

            with torch.no_grad():
                dec_out = model.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attn_mask,
                    encoder_hidden_states=encoder_dummy,
                )
            hidden_batch = dec_out.last_hidden_state.float().cpu().numpy()

            for b, utt_id in enumerate(batch_ids):
                hidden = hidden_batch[b]
                om     = offset_mapping[b]
                for word_idx in utt_to_words[utt_id]:
                    if word_idx in completed_words:
                        continue
                    rec    = word_records[word_idx]
                    char_s = rec.get("char_start", -1)
                    char_e = rec.get("char_end",   -1)
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

        except Exception as e:
            errors += 1
            logger.warning(f"Skipping batch at {batch_start}: {e}")

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    if errors:
        logger.warning(f"{model_name}: {errors} errors")

    pairs_sorted = sorted(word_embeddings_list, key=lambda x: x[0])
    if not pairs_sorted:
        raise RuntimeError(f"No embeddings collected for {model_name}")
    D   = pairs_sorted[0][1].shape[0]
    arr = np.zeros((N, D), dtype=np.float32)
    for idx, emb in pairs_sorted:
        arr[idx] = emb
    return arr


# ---------------------------------------------------------------------------
# CKA — two variants
# ---------------------------------------------------------------------------

def _hsic1_batch(K: np.ndarray, L: np.ndarray) -> float:
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


def _cka_per_group(X, Y, word_ids):
    """Per-group CKA for cross-speaker data (MCV)."""
    valid    = (np.linalg.norm(X, axis=1) > 1e-10) & (np.linalg.norm(Y, axis=1) > 1e-10)
    X, Y     = X[valid], Y[valid]
    word_ids = word_ids[valid]
    hsic_xy, hsic_xx, hsic_yy = [], [], []
    for wid in np.unique(word_ids):
        idx = np.where(word_ids == wid)[0]
        if len(idx) < 4:
            continue
        Xg = X[idx].astype(np.float64)
        Yg = Y[idx].astype(np.float64)
        Xg /= np.linalg.norm(Xg, axis=1, keepdims=True) + 1e-10
        Yg /= np.linalg.norm(Yg, axis=1, keepdims=True) + 1e-10
        K = Xg @ Xg.T; L = Yg @ Yg.T
        hsic_xy.append(_hsic1_batch(K, L))
        hsic_xx.append(_hsic1_batch(K, K))
        hsic_yy.append(_hsic1_batch(L, L))
    if not hsic_xy:
        return 0.0, 0.0
    mean_xy = float(np.mean(hsic_xy))
    denom   = np.sqrt(max(float(np.mean(hsic_xx)), 0.0) * max(float(np.mean(hsic_yy)), 0.0))
    score   = float(mean_xy / denom) if denom > 1e-10 else 0.0
    per_group = [hxy / np.sqrt(max(hxx, 0.0) * max(hyy, 0.0))
                 if np.sqrt(max(hxx, 0.0) * max(hyy, 0.0)) > 1e-10 else 0.0
                 for hxy, hxx, hyy in zip(hsic_xy, hsic_xx, hsic_yy)]
    ci95 = 1.96 * float(np.std(per_group)) / np.sqrt(len(per_group)) if len(per_group) > 1 else 0.0
    return score, ci95


def _cka_minibatch(X, Y, batch_size=MINIBATCH_SIZE, seed=MINIBATCH_SEED):
    """Random-minibatch CKA for single-speaker data (LJSpeech)."""
    valid = (np.linalg.norm(X, axis=1) > 1e-10) & (np.linalg.norm(Y, axis=1) > 1e-10)
    X, Y  = X[valid], Y[valid]
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X))
    X, Y  = X[idx], Y[idx]
    hsic_xy, hsic_xx, hsic_yy = [], [], []
    for start in range(0, len(X) - batch_size + 1, batch_size):
        Xb = X[start : start + batch_size].astype(np.float64)
        Yb = Y[start : start + batch_size].astype(np.float64)
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-10
        Yb /= np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-10
        K = Xb @ Xb.T; L = Yb @ Yb.T
        hsic_xy.append(_hsic1_batch(K, L))
        hsic_xx.append(_hsic1_batch(K, K))
        hsic_yy.append(_hsic1_batch(L, L))
    if not hsic_xy:
        return 0.0, 0.0
    mean_xy = float(np.mean(hsic_xy))
    denom   = np.sqrt(max(float(np.mean(hsic_xx)), 0.0) * max(float(np.mean(hsic_yy)), 0.0))
    score   = float(mean_xy / denom) if denom > 1e-10 else 0.0
    per_batch = [hxy / np.sqrt(max(hxx, 0.0) * max(hyy, 0.0))
                 if np.sqrt(max(hxx, 0.0) * max(hyy, 0.0)) > 1e-10 else 0.0
                 for hxy, hxx, hyy in zip(hsic_xy, hsic_xx, hsic_yy)]
    ci95 = 1.96 * float(np.std(per_batch)) / np.sqrt(len(per_batch)) if len(per_batch) > 1 else 0.0
    return score, ci95


def compute_pairwise_cka(
    embeddings: dict[str, np.ndarray],
    names: list[str],
    cross_speaker: bool,
    word_ids: np.ndarray | None = None,
) -> np.ndarray:
    n    = len(names)
    mat  = np.eye(n)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in tqdm(pairs, desc="CKA pairs", unit="pair"):
        if cross_speaker and word_ids is not None:
            score, _ = _cka_per_group(embeddings[names[i]], embeddings[names[j]], word_ids)
        else:
            score, _ = _cka_minibatch(embeddings[names[i]], embeddings[names[j]])
        mat[i, j] = mat[j, i] = score
        logger.info(f"  CKA({names[i]}, {names[j]}) = {score:.4f}")
    return mat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    names: list[str],
    plots_dir: Path,
    prefix: str,
    tag: str = "deaf_decoder_cka",
):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    n    = len(names)
    groups = [_model_group(nm) for nm in names]

    cmap = LinearSegmentedColormap.from_list("cka", ["#FFFFFF", "#A5D6A7", "#2E7D32"], N=256)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(8, n * 0.5)))
    im = ax.imshow(cka_matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA  (0 = no similarity,  1 = identical)", fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = cka_matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    # Group colour bar on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(n))
    ax2.set_xticklabels([""] * n)
    for tick, name in zip(ax2.get_xticklabels(), names):
        tick.set_color(GROUP_COLORS.get(_model_group(name), "#000000"))
    # Draw thin coloured rectangles above each column label
    for xi, name in enumerate(names):
        color = GROUP_COLORS.get(_model_group(name), "#AAAAAA")
        ax.add_patch(plt.Rectangle(
            (xi - 0.5, -1.5), 1, 0.5,
            color=color, clip_on=False, transform=ax.transData,
        ))

    # Legend for groups
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()
                      if g in groups]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              title="Model family", title_fontsize=8,
              bbox_to_anchor=(1.18, 0), borderaxespad=0)

    ax.set_title("Whisper Deaf-Decoder Analysis — Linear CKA", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()

    out_path = plots_dir / f"{prefix}{tag}_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved heatmap → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=".", type=Path)
    p.add_argument("--dataset", default="mcv", choices=list(DATASET_CONFIGS),
                   help="mcv (cross-speaker CKA) or word (LJSpeech, minibatch CKA)")
    p.add_argument("--dec_batch_size", default=32, type=int)
    p.add_argument("--skip_extraction", action="store_true",
                   help="Skip deaf-decoder extraction; use cached PKL files only")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    root = Path(args.root_dir)
    cfg  = DATASET_CONFIGS[args.dataset]

    data_dir  = root / cfg["data_dir"]
    plots_dir = root / cfg["plots_dir"] / "deaf_decoder_cka"
    plots_dir.mkdir(parents=True, exist_ok=True)
    prefix     = cfg["prefix"]
    pkl_prefix = cfg["pkl_prefix"]

    # ── Load word records ─────────────────────────────────────────────────────
    wr_path = data_dir / cfg["word_records"]
    if not wr_path.exists():
        logger.error(f"Word records not found: {wr_path}. Run word-level analysis first.")
        sys.exit(1)
    with open(wr_path) as f:
        word_records = json.load(f)
    logger.info(f"Loaded {len(word_records):,} word records")

    # ── Build word_ids (same-word/sentence/position grouping) ─────────────────
    all_types: dict = {}
    ids = []
    for rec in word_records:
        key = (rec["word"], rec["sentence"], rec.get("char_start", -1))
        if key not in all_types:
            all_types[key] = len(all_types)
        ids.append(all_types[key])
    word_ids = np.array(ids, dtype=np.int32)
    logger.info(f"Word type groups: {len(all_types):,}")

    # ── Load existing cached embeddings ───────────────────────────────────────
    embeddings: dict[str, np.ndarray] = {}
    pkl_pattern = re.compile(rf"^{re.escape(pkl_prefix)}word_embeddings_(.+)\.pkl$")
    for pkl_path in sorted(data_dir.glob(f"{pkl_prefix}word_embeddings_*.pkl")):
        m = pkl_pattern.match(pkl_path.name)
        if not m:
            continue
        model_name = m.group(1)
        logger.info(f"Loading cached embeddings: {model_name}")
        with open(pkl_path, "rb") as f:
            embeddings[model_name] = pickle.load(f)
        logger.info(f"  Shape: {embeddings[model_name].shape}")

    if not embeddings:
        logger.error(f"No cached embedding PKLs found in {data_dir}. "
                     "Run the word-level analysis script first.")
        sys.exit(1)

    # ── Determine which deaf decoders to extract ──────────────────────────────
    # Extract a deaf decoder for every normal decoder that is either already
    # cached or present in WHISPER_DEC_HF_IDS
    deaf_needed = [nm for nm in WHISPER_DEC_HF_IDS if nm in embeddings]

    # ── Load audio (needed only for extraction) ───────────────────────────────
    utterances = {}
    need_audio = not args.skip_extraction and any(
        f"{nm}-deaf" not in embeddings and
        not (data_dir / f"{prefix}word_embeddings_{nm}-deaf.pkl").exists()
        for nm in deaf_needed
    )
    if need_audio:
        utterances = load_utterances(cfg["loader"], data_dir, cfg["audio_subdir"])

    device = get_device()

    # ── Extract / load deaf decoder embeddings ────────────────────────────────
    for base_name in deaf_needed:
        deaf_name  = f"{base_name}-deaf"
        cache_path = data_dir / f"{pkl_prefix}word_embeddings_{deaf_name}.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached deaf decoder: {deaf_name}")
            with open(cache_path, "rb") as f:
                embeddings[deaf_name] = pickle.load(f)
            logger.info(f"  Shape: {embeddings[deaf_name].shape}")
            continue

        if args.skip_extraction:
            logger.warning(f"--skip_extraction but no cache for {deaf_name}, skipping")
            continue

        emb = extract_deaf_decoder_embeddings(
            deaf_name, WHISPER_DEC_HF_IDS[base_name],
            word_records, utterances, device, batch_size=args.dec_batch_size,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(emb, f)
        logger.info(f"Cached {deaf_name} → {cache_path}")
        embeddings[deaf_name] = emb

    # ── Apply valid mask ──────────────────────────────────────────────────────
    N = len(word_records)
    valid_mask = np.ones(N, dtype=bool)
    for emb in embeddings.values():
        if emb.shape[0] == N:
            valid_mask &= np.linalg.norm(emb, axis=1) > 1e-10
    for nm in list(embeddings):
        if embeddings[nm].shape[0] == N:
            embeddings[nm] = embeddings[nm][valid_mask]
        else:
            logger.warning(f"Dropping {nm}: shape mismatch ({embeddings[nm].shape[0]} != {N})")
            del embeddings[nm]
    word_ids_masked = word_ids[valid_mask]

    # ── Order models for heatmap ──────────────────────────────────────────────
    group_order = ["Encoder", "Decoder", "Deaf decoder", "Audio (other)", "LLM"]
    size_order  = ["base", "small", "medium", "large-v1", "large-v2", "large"]

    def sort_key(name):
        g = group_order.index(_model_group(name))
        for si, sz in enumerate(size_order):
            if sz in name:
                return (g, si, name)
        return (g, 99, name)

    names = sorted(embeddings.keys(), key=sort_key)
    logger.info(f"Models in comparison ({len(names)}): {names}")

    # ── Compute pairwise CKA ──────────────────────────────────────────────────
    cka_matrix = compute_pairwise_cka(
        embeddings, names,
        cross_speaker=cfg["cross_speaker"],
        word_ids=word_ids_masked if cfg["cross_speaker"] else None,
    )

    # ── Save CKA matrix ───────────────────────────────────────────────────────
    cka_save = {"names": names, "matrix": cka_matrix.tolist()}
    cka_json = plots_dir / f"{prefix}deaf_decoder_cka_matrix.json"
    with open(cka_json, "w") as f:
        json.dump(cka_save, f, indent=2)
    logger.info(f"Saved CKA matrix → {cka_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_cka_heatmap(cka_matrix, names, plots_dir, prefix)
    logger.info("Done.")


if __name__ == "__main__":
    main()
