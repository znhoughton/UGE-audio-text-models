#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Word embeddings live in the repo's own WordData/; phone embeddings in /dpluth-data/PhoneData/
WORD_EMBEDDINGS_DIR="."
PHONE_EMBEDDINGS_DIR="/dpluth-data"
WORD_OUT="whisper_kaldi_llm_projection/word"
PHONE_OUT="whisper_kaldi_llm_projection/phone"
ANALYSIS="whisper_kaldi_llm_projection/projection_analysis.py"

mkdir -p "$WORD_OUT" "$PHONE_OUT"

# All 12 Whisper models (encoder + decoder, every size). One process loads
# Kaldi/LLM once and iterates through the full list. Phone OLMo is ~14.5 GB,
# so each granularity runs as a single sequential process (never parallel copies).
WHISPER_MODELS=(
    whisper-base-enc whisper-small-enc whisper-medium-enc
    whisper-large-enc whisper-large-v2-enc whisper-large-v1-enc
    whisper-base-dec whisper-small-dec whisper-medium-dec
    whisper-large-dec whisper-large-v2-dec whisper-large-v1-dec
)

# ---------------------------------------------------------------------------
# Word-level
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Starting word-level (sequential, all models)..."
python "$ANALYSIS" \
    --granularity word \
    --embeddings_dir "$WORD_EMBEDDINGS_DIR" \
    --output_dir "$WORD_OUT" \
    --whisper_models "${WHISPER_MODELS[@]}"
echo "[$(date '+%H:%M:%S')] Word-level complete."

# ---------------------------------------------------------------------------
# Phone-level
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Starting phone-level (sequential, all models)..."
python "$ANALYSIS" \
    --granularity phone \
    --embeddings_dir "$PHONE_EMBEDDINGS_DIR" \
    --output_dir "$PHONE_OUT" \
    --whisper_models "${WHISPER_MODELS[@]}"
echo "[$(date '+%H:%M:%S')] Phone-level complete."

echo "[$(date '+%H:%M:%S')] All complete."
