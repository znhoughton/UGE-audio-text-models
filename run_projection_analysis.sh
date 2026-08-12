#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EMBEDDINGS_DIR="/dpluth-data"
WORD_OUT="whisper_kaldi_llm_projection/word"
PHONE_OUT="whisper_kaldi_llm_projection/phone"
ANALYSIS="whisper_kaldi_llm_projection/projection_analysis.py"

mkdir -p "$WORD_OUT" "$PHONE_OUT"

run_group() {
    local label="$1"
    shift
    local models=("$@")

    echo "[$(date '+%H:%M:%S')] Starting group: $label"

    python "$ANALYSIS" \
        --granularity word \
        --embeddings_dir "$EMBEDDINGS_DIR" \
        --output_dir "$WORD_OUT" \
        --whisper_models "${models[@]}" \
    && \
    python "$ANALYSIS" \
        --granularity phone \
        --embeddings_dir "$EMBEDDINGS_DIR" \
        --output_dir "$PHONE_OUT" \
        --whisper_models "${models[@]}"

    echo "[$(date '+%H:%M:%S')] Finished group: $label"
}

# Group 1: encoders base–large
run_group "encoders-base-large" \
    whisper-base-enc whisper-small-enc whisper-medium-enc whisper-large-enc &
PID1=$!

# Group 2: encoders large-v2/v1 + decoders base/small
run_group "encoders-largev2v1-decoders-base-small" \
    whisper-large-v2-enc whisper-large-v1-enc whisper-base-dec whisper-small-dec &
PID2=$!

# Group 3: decoders medium–large-v1
run_group "decoders-medium-large" \
    whisper-medium-dec whisper-large-dec whisper-large-v2-dec whisper-large-v1-dec &
PID3=$!

echo "All 3 groups running in parallel (PIDs: $PID1 $PID2 $PID3)"
echo "Word-level finishes first in each group, then phone-level starts."

wait $PID1 && echo "Group 1 done" || echo "Group 1 FAILED (exit $?)"
wait $PID2 && echo "Group 2 done" || echo "Group 2 FAILED (exit $?)"
wait $PID3 && echo "Group 3 done" || echo "Group 3 FAILED (exit $?)"

echo "[$(date '+%H:%M:%S')] All groups complete."
