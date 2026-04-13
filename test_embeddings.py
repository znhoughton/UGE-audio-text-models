#!/usr/bin/env python3
"""
Embedding extraction smoke test.

Loads a small batch of real LibriSpeech audio and text, runs each model's
full extraction pipeline (same code path as the main script), and validates
the output embeddings are finite, correctly shaped, and meaningfully different
from random noise.

Runs each model on 16 samples only — fast enough to complete in a few minutes
and catches NaN/shape/device bugs before committing to a multi-hour full run.

Usage:
    python test_embeddings.py                    # test all models
    python test_embeddings.py --models whisper-base parakeet-ctc-0.6b
    python test_embeddings.py --n_samples 32     # larger batch
"""

import argparse
import gc
import os
import sys
import numpy as np

# Point HF cache to workspace before any HF imports
os.environ.setdefault("HF_HOME", "/workspace/huggingface_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/workspace/huggingface_cache/datasets")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    ParakeetForCTC,
    WhisperModel,
    WhisperProcessor,
)

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30
MAX_TEXT_TOKENS = 512

MODELS = {
    "whisper-base": {
        "hf_id": "openai/whisper-base",
        "modality": "audio",
        "expected_dim": 512,
    },
    "parakeet-ctc-0.6b": {
        "hf_id": "nvidia/parakeet-ctc-0.6b",
        "modality": "audio-parakeet",
        "expected_dim": 1024,
    },
    "babylm-125m": {
        "hf_id": "znhoughton/opt-babylm-125m-20eps-seed964",
        "modality": "text",
        "expected_dim": 768,
    },
    "babylm-1.3b": {
        "hf_id": "znhoughton/opt-babylm-1.3b-20eps-seed964",
        "modality": "text",
        "expected_dim": 2048,
    },
    "olmo-7b": {
        "hf_id": "allenai/OLMo-2-1124-7B",
        "modality": "text",
        "expected_dim": 4096,
    },
    "pythia-6.9b": {
        "hf_id": "EleutherAI/pythia-6.9b",
        "modality": "text",
        "expected_dim": 4096,
    },
}


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label, condition, detail=""):
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")
    return condition


def release_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def validate_embeddings(emb: np.ndarray, model_name: str, n_samples: int, expected_dim: int) -> bool:
    """Run all validation checks on an embedding matrix. Returns True if all pass."""
    all_pass = True

    all_pass &= check(
        "Shape is (n_samples, hidden_dim)",
        emb.shape == (n_samples, expected_dim),
        f"got {emb.shape}, expected ({n_samples}, {expected_dim})"
    )
    all_pass &= check(
        "All values are finite (no NaN or Inf)",
        np.isfinite(emb).all(),
        f"non-finite count: {(~np.isfinite(emb)).sum()}"
    )
    all_pass &= check(
        "Embeddings are not all zero",
        not np.allclose(emb, 0),
        f"mean abs value: {np.abs(emb).mean():.4f}"
    )
    all_pass &= check(
        "Embeddings vary across samples (not all identical)",
        emb.std(axis=0).mean() > 1e-6,
        f"mean std across samples: {emb.std(axis=0).mean():.6f}"
    )

    # Check that different samples actually differ from each other
    # Cosine similarity between first two samples should not be 1.0
    a = emb[0] / (np.linalg.norm(emb[0]) + 1e-10)
    b = emb[1] / (np.linalg.norm(emb[1]) + 1e-10)
    cos_sim = float(a @ b)
    all_pass &= check(
        "Different samples produce different embeddings",
        cos_sim < 0.9999,
        f"cosine similarity between sample 0 and 1: {cos_sim:.6f}"
    )

    # Norm distribution check — embeddings shouldn't all have identical norms
    norms = np.linalg.norm(emb, axis=1)
    all_pass &= check(
        "Embedding norms vary across samples",
        norms.std() > 1e-6,
        f"norm mean={norms.mean():.3f}, std={norms.std():.3f}"
    )

    return all_pass


def test_whisper(samples, n_samples, device):
    section("Whisper-base")
    model_id = MODELS["whisper-base"]["hf_id"]
    expected_dim = MODELS["whisper-base"]["expected_dim"]

    print(f"  Loading {model_id}…")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()

    audio_arrays = []
    for s in samples:
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        sr = s["audio"]["sampling_rate"]
        if sr != SAMPLE_RATE:
            audio = audio[:: int(sr / SAMPLE_RATE)]
        audio = audio[: MAX_AUDIO_SECONDS * SAMPLE_RATE]
        audio_arrays.append(audio)

    inputs = processor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    features = inputs.input_features.to(device, dtype=torch.float16)

    print(f"  Input shape: {tuple(features.shape)}")
    check("Mel features are (batch, 80, 3000)", features.shape == (n_samples, 80, 3000),
          f"got {tuple(features.shape)}")

    with torch.no_grad():
        enc = model.encoder(features)
        emb = enc.last_hidden_state.mean(dim=1).float().cpu().numpy()

    print(f"  Output embedding shape: {emb.shape}")
    result = validate_embeddings(emb, "whisper-base", n_samples, expected_dim)

    del model
    release_vram()
    return result


def test_parakeet(samples, n_samples, device):
    section("Parakeet-CTC-0.6b")
    model_id = MODELS["parakeet-ctc-0.6b"]["hf_id"]
    expected_dim = MODELS["parakeet-ctc-0.6b"]["expected_dim"]

    print(f"  Loading {model_id}…")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = ParakeetForCTC.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # fp32 — FastConformer NaNs in fp16
        device_map="auto",
        max_memory={0: "75GiB", 1: "75GiB"},
    )
    model.eval()

    first_device = next(iter(model.hf_device_map.values())) if hasattr(model, "hf_device_map") else device

    audio_arrays = []
    for s in samples:
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        sr = s["audio"]["sampling_rate"]
        if sr != SAMPLE_RATE:
            audio = audio[:: int(sr / SAMPLE_RATE)]
        audio = audio[: MAX_AUDIO_SECONDS * SAMPLE_RATE]
        audio_arrays.append(audio)

    inputs = feature_extractor(audio_arrays, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=True)
    input_key = next(k for k in inputs.keys() if "mask" not in k)
    input_tensor = inputs[input_key].to(first_device, dtype=torch.float32)

    print(f"  Feature extractor key: '{input_key}', input shape: {tuple(input_tensor.shape)}")

    with torch.no_grad():
        out = model(input_tensor, output_hidden_states=True)
        emb = out.hidden_states[-1].mean(dim=1).float().cpu().numpy()

    print(f"  n_hidden_layers: {len(out.hidden_states)}")
    print(f"  Output embedding shape: {emb.shape}")
    result = validate_embeddings(emb, "parakeet-ctc-0.6b", n_samples, expected_dim)

    del model
    release_vram()
    return result


def test_lm(model_name, texts, n_samples, device):
    section(f"LM: {model_name}")
    cfg = MODELS[model_name]
    model_id = cfg["hf_id"]
    expected_dim = cfg["expected_dim"]

    print(f"  Loading {model_id}…")
    load_kwargs = dict(
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: "75GiB", 1: "75GiB"},
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)
    model.eval()

    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
    else:
        first_device = next(model.parameters()).device
    print(f"  First layer device: {first_device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_TEXT_TOKENS)
    input_ids = enc["input_ids"].to(first_device)
    attention_mask = enc["attention_mask"].to(first_device)

    print(f"  Tokenized shape: {tuple(input_ids.shape)}")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
        hidden = out.hidden_states[-1].float()

    mask = attention_mask.unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    emb = pooled.cpu().numpy()

    print(f"  n_hidden_layers: {len(out.hidden_states)}")
    print(f"  Output embedding shape: {emb.shape}")
    result = validate_embeddings(emb, model_name, n_samples, expected_dim)

    del model
    release_vram()
    return result


def main():
    parser = argparse.ArgumentParser(description="Smoke test embedding extraction")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Which models to test (default: all)")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of samples to test with (default: 16)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free = torch.cuda.mem_get_info(i)[0] / 1024**3
            print(f"  GPU {i}: {props.name}  {props.total_memory/1024**3:.0f}GB total  {free:.1f}GB free")

    # Load a small real dataset batch
    section("Loading test data")
    print(f"  Loading {args.n_samples} samples from LibriSpeech test-clean…")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test",
                      trust_remote_code=True)
    samples = [ds[i] for i in range(args.n_samples)]
    texts = [s["text"].strip() for s in samples]
    print(f"  Loaded {len(samples)} samples")
    print(f"  Example text: '{texts[0][:60]}'")
    print(f"  Example audio length: {len(samples[0]['audio']['array'])} samples "
          f"({len(samples[0]['audio']['array'])/SAMPLE_RATE:.1f}s)")

    # Run tests
    results = {}

    for model_name in args.models:
        modality = MODELS[model_name]["modality"]
        try:
            if modality == "audio":
                results[model_name] = test_whisper(samples, args.n_samples, device)
            elif modality == "audio-parakeet":
                results[model_name] = test_parakeet(samples, args.n_samples, device)
            else:
                results[model_name] = test_lm(model_name, texts, args.n_samples, device)
        except Exception as e:
            import traceback
            print(f"\n  ✗ FAIL  {model_name} raised an exception: {e}")
            traceback.print_exc()
            results[model_name] = False
            release_vram()

    # Summary
    section("Summary")
    all_pass = True
    for model_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {model_name}")
        all_pass = all_pass and passed

    print()
    if all_pass:
        print("  All models passed — safe to run the full extraction.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  FAILED models: {failed}")
        print("  Fix these before running the full script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
