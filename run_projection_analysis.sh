#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHONE_EMBEDDINGS_DIR="/dpluth-data"
PHONE_OUT="whisper_kaldi_llm_projection/phone"
ANALYSIS="whisper_kaldi_llm_projection/projection_analysis.py"

mkdir -p "$PHONE_OUT"

# ---------------------------------------------------------------------------
# Phone-level — single sequential process
# Phone OLMo alone is ~14.5 GB; running 3 parallel copies OOMs the pod.
# One process loads OLMo/Kaldi once and iterates through all Whisper models.
# ---------------------------------------------------------------------------

echo "[$(date '+%H:%M:%S')] Starting phone-level (sequential, all models)..."
python "$ANALYSIS" \
    --granularity phone \
    --embeddings_dir "$PHONE_EMBEDDINGS_DIR" \
    --output_dir "$PHONE_OUT" \
    --whisper_models \
        whisper-base-enc whisper-small-enc whisper-medium-enc \
        whisper-large-enc whisper-large-v2-enc whisper-large-v1-enc \
        whisper-base-dec whisper-small-dec whisper-medium-dec \
        whisper-large-dec whisper-large-v2-dec whisper-large-v1-dec

echo "[$(date '+%H:%M:%S')] All complete."
