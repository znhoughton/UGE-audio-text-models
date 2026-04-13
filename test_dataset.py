#!/usr/bin/env python3
"""
Quick sanity checks before running the full representation analysis.

Tests:
  1. Single split audio loading via ds[i]
  2. select() on single split
  3. select() on concatenated dataset (known to fail — confirms the bug)
  4. iter() on concatenated dataset (our fix)
  5. Whisper processor with a real audio array
  6. That mel features are exactly 3000 frames after padding

Usage:
    python test_dataset.py
"""

import sys
import numpy as np

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30

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

# ---------------------------------------------------------------------------
# 1. Load two small splits to test concatenation
# ---------------------------------------------------------------------------
section("1. Loading dataset splits")
from datasets import load_dataset, concatenate_datasets

print("  Loading test split (clean)…")
ds_test = load_dataset(
    "openslr/librispeech_asr", "clean",
    split="test", trust_remote_code=True
)
print(f"  test split: {len(ds_test):,} samples")

print("  Loading validation split (clean)…")
ds_val = load_dataset(
    "openslr/librispeech_asr", "clean",
    split="validation", trust_remote_code=True
)
print(f"  validation split: {len(ds_val):,} samples")

# ---------------------------------------------------------------------------
# 2. Single split — direct indexing
# ---------------------------------------------------------------------------
section("2. Single split — direct indexing")
sample = ds_test[0]
audio = np.array(sample["audio"]["array"], dtype=np.float32)
check(
    "Audio array is a numpy-compatible float array",
    len(audio) > 1000,
    f"length={len(audio)}, dtype={audio.dtype}"
)
check(
    "Sample rate is 16000",
    sample["audio"]["sampling_rate"] == SAMPLE_RATE,
    f"sr={sample['audio']['sampling_rate']}"
)
check(
    "Text field is a non-empty string",
    isinstance(sample["text"], str) and len(sample["text"]) > 0,
    f"text='{sample['text'][:50]}'"
)

# ---------------------------------------------------------------------------
# 3. Single split — select()
# ---------------------------------------------------------------------------
section("3. Single split — select()")
batch = ds_test.select([0, 1, 2])
all_ok = True
for i, s in enumerate(batch):
    a = np.array(s["audio"]["array"], dtype=np.float32)
    ok = len(a) > 1000
    check(f"select() sample {i} audio length > 1000", ok, f"length={len(a)}")
    all_ok = all_ok and ok

# ---------------------------------------------------------------------------
# 4. Concatenated dataset — select() (expected to fail)
# ---------------------------------------------------------------------------
section("4. Concatenated dataset — select() [expect FAIL]")
ds_concat = concatenate_datasets([ds_test, ds_val])
print(f"  Concatenated size: {len(ds_concat):,}")

batch_select = ds_concat.select([0, 1, 2])
select_ok = True
for i, s in enumerate(batch_select):
    a = np.array(s["audio"]["array"], dtype=np.float32)
    ok = len(a) > 1000
    check(
        f"select() on concat sample {i} audio length > 1000",
        ok,
        f"length={len(a)} {'← BAD (raw bytes)' if not ok else ''}"
    )
    select_ok = select_ok and ok

if not select_ok:
    print("  → Confirmed: select() on concatenated dataset returns raw bytes, not decoded audio.")
else:
    print("  → select() worked fine on concat — behaviour may have changed in this datasets version.")

# ---------------------------------------------------------------------------
# 5. Concatenated dataset — iter() (our fix)
# ---------------------------------------------------------------------------
section("5. Concatenated dataset — iter() [our fix]")
batch_iter_raw = next(ds_concat.iter(batch_size=4))

# Probe the structure so we know exactly what iter() returns
print(f"  iter() batch keys: {list(batch_iter_raw.keys())}")
print(f"  type of batch['audio']: {type(batch_iter_raw['audio'])}")
if isinstance(batch_iter_raw["audio"], list):
    print(f"  batch['audio'] is a LIST of {len(batch_iter_raw['audio'])} items")
    print(f"  type of batch['audio'][0]: {type(batch_iter_raw['audio'][0])}")
    if isinstance(batch_iter_raw["audio"][0], dict):
        print(f"  batch['audio'][0] keys: {list(batch_iter_raw['audio'][0].keys())}")
    # iter() returns list of dicts: [{"array": ..., "sampling_rate": ...}, ...]
    iter_ok = True
    for i, audio_dict in enumerate(batch_iter_raw["audio"]):
        a = np.array(audio_dict["array"], dtype=np.float32)
        sr = audio_dict["sampling_rate"]
        ok = len(a) > 1000
        check(f"iter() on concat sample {i} audio length > 1000", ok, f"length={len(a)}, sr={sr}")
        iter_ok = iter_ok and ok
elif isinstance(batch_iter_raw["audio"], dict):
    print(f"  batch['audio'] is a DICT with keys: {list(batch_iter_raw['audio'].keys())}")
    # iter() returns dict of lists: {"array": [...], "sampling_rate": [...]}
    iter_ok = True
    for i, (arr, sr) in enumerate(zip(batch_iter_raw["audio"]["array"], batch_iter_raw["audio"]["sampling_rate"])):
        a = np.array(arr, dtype=np.float32)
        ok = len(a) > 1000
        check(f"iter() on concat sample {i} audio length > 1000", ok, f"length={len(a)}, sr={sr}")
        iter_ok = iter_ok and ok

# Store for use in Whisper test below — works regardless of structure
audio_samples_for_whisper = []
for audio_item in batch_iter_raw["audio"][:3]:
    if isinstance(audio_item, dict):
        arr = np.array(audio_item["array"], dtype=np.float32)
        sr = audio_item["sampling_rate"]
    else:
        # fallback if it's already an array
        arr = np.array(audio_item, dtype=np.float32)
        sr = SAMPLE_RATE
    audio_samples_for_whisper.append((arr, sr))

# ---------------------------------------------------------------------------
# 6. Whisper processor — probe what padding args actually do
# ---------------------------------------------------------------------------
section("6. Whisper processor padding")
try:
    from transformers import WhisperProcessor
    print("  Loading WhisperProcessor…")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")

    audio_arrays = []
    for arr, sr in audio_samples_for_whisper:
        a = arr.copy()
        if sr != SAMPLE_RATE:
            a = a[:: int(sr / SAMPLE_RATE)]
        a = a[: MAX_AUDIO_SECONDS * SAMPLE_RATE]
        audio_arrays.append(a)

    print(f"  Raw audio lengths (samples): {[len(a) for a in audio_arrays]}")

    # Probe 1: no padding args — processor default behaviour
    inp1 = processor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    print(f"  No padding args:                      shape={tuple(inp1.input_features.shape)}")

    # Probe 2: current code — max_length=3000 (probably wrong units)
    inp2 = processor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                     padding="max_length", max_length=3000, truncation=True)
    print(f"  padding='max_length', max_length=3000: shape={tuple(inp2.input_features.shape)}")

    # Probe 3: max_length in raw samples (30s * 16000 = 480000)
    inp3 = processor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                     padding="max_length", max_length=480000, truncation=True)
    print(f"  padding='max_length', max_length=480000: shape={tuple(inp3.input_features.shape)}")

    # The correct approach — use default (no padding args) since processor
    # always outputs fixed 3000 frames regardless
    shape = inp1.input_features.shape
    check("Batch dimension matches", shape[0] == len(audio_arrays), f"shape={tuple(shape)}")
    check("Default processor gives exactly 3000 mel frames", shape[2] == 3000, f"mel_length={shape[2]}")
    check("Mel frequency bins are 80", shape[1] == 80, f"n_mels={shape[1]}")

    if shape[2] == 3000:
        print("\n  → CONCLUSION: Use processor() with NO padding args.")
        print("    The WhisperProcessor always outputs exactly 3000 frames by default.")
        print("    'padding' and 'max_length' args operate on raw audio samples, not mel frames.")
    else:
        print(f"\n  → Default gave {shape[2]} frames — investigate further.")

except Exception as e:
    import traceback
    print(f"  ✗ FAIL  Whisper processor test: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
section("Summary")
print("  If all iter() tests passed and select() on concat failed,")
print("  the fix in representation_analysis.py is correct and safe to run.")
print()
print("  Key finding:")
if iter_ok and not select_ok:
    print("  ✓ iter() works correctly, select() on concat is broken — fix is validated.")
elif iter_ok and select_ok:
    print("  Both iter() and select() work — may be a datasets version difference.")
    print("  iter() is still the safer choice.")
else:
    print("  ✗ iter() also failing — deeper issue, investigate before running main script.")
