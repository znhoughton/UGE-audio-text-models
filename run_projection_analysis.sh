#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WORD_EMBEDDINGS_DIR="."
PHONE_EMBEDDINGS_DIR="/dpluth-data"
WORD_OUT="whisper_kaldi_llm_projection/word"
PHONE_OUT="whisper_kaldi_llm_projection/phone"
ANALYSIS="whisper_kaldi_llm_projection/projection_analysis.py"

mkdir -p "$WORD_OUT" "$PHONE_OUT"

# ---------------------------------------------------------------------------
# Phase 1: word-level — 3 parallel groups
# Word embeddings are small enough that 3×OLMo fits in RAM.
# ---------------------------------------------------------------------------

run_word_group() {
    local label="$1"
    shift
    local models=("$@")
    echo "[$(date '+%H:%M:%S')] Starting word group: $label"
    python "$ANALYSIS" \
        --granularity word \
        --embeddings_dir "$WORD_EMBEDDINGS_DIR" \
        --output_dir "$WORD_OUT" \
        --whisper_models "${models[@]}"
    echo "[$(date '+%H:%M:%S')] Finished word group: $label"
}

run_word_group "encoders-base-large" \
    whisper-base-enc whisper-small-enc whisper-medium-enc whisper-large-enc &
PID1=$!

run_word_group "encoders-largev2v1-decoders-base-small" \
    whisper-large-v2-enc whisper-large-v1-enc whisper-base-dec whisper-small-dec &
PID2=$!

run_word_group "decoders-medium-large" \
    whisper-medium-dec whisper-large-dec whisper-large-v2-dec whisper-large-v1-dec &
PID3=$!

echo "Word-level: 3 groups running in parallel (PIDs: $PID1 $PID2 $PID3)"
wait $PID1 && echo "Word group 1 done" || echo "Word group 1 FAILED (exit $?)"
wait $PID2 && echo "Word group 2 done" || echo "Word group 2 FAILED (exit $?)"
wait $PID3 && echo "Word group 3 done" || echo "Word group 3 FAILED (exit $?)"
echo "[$(date '+%H:%M:%S')] Word-level complete."

# ---------------------------------------------------------------------------
# Phase 2: phone-level — single sequential process
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
