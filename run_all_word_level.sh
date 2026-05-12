#!/usr/bin/env bash
# Run all three word-level representation analyses sequentially.
# Edit the variables below, then: bash run_all_word_level.sh
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

# Path to your local Mozilla Common Voice English directory
# (must contain validated.tsv and a clips/ folder)
MCV_SOURCE_DIR="/opt/modeling/dpluth/data/mcv/en"

# Output root — all Data/ and Plots/ folders will be created here
ROOT_DIR="/opt/modeling/zhoughton/misc/UGE-audio-text-models"

# MCV sampling settings
MIN_SPEAKERS=250
N_WORDS=250000

# Set to "--skip_download", "--skip_mfa", or "--skip_extraction" to resume
# a partial run. Leave empty ("") to run the full pipeline.
SKIP_DOWNLOAD=""
SKIP_MFA=""
SKIP_EXTRACTION=""

# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_stage() {
    local name="$1"; shift
    log "========== START: $name =========="
    local t0=$SECONDS
    "$@"
    local elapsed=$(( SECONDS - t0 ))
    log "========== DONE:  $name  (${elapsed}s) =========="
    echo
}

run_stage "LJSpeech word-level analysis" \
    python "$SCRIPT_DIR/word_level_analysis.py" \
        --root_dir "$ROOT_DIR" \
        $SKIP_EXTRACTION

run_stage "MLS word-level analysis" \
    python "$SCRIPT_DIR/mls_word_level_analysis.py" \
        --root_dir "$ROOT_DIR" \
        $SKIP_DOWNLOAD \
        $SKIP_MFA \
        $SKIP_EXTRACTION

run_stage "MCV cross-speaker word-level analysis" \
    python "$SCRIPT_DIR/mcv_word_level_analysis_cross_speakers.py" \
        --root_dir "$ROOT_DIR" \
        --mcv_source_dir "$MCV_SOURCE_DIR" \
        --min_speakers "$MIN_SPEAKERS" \
        --n_words "$N_WORDS" \
        $SKIP_DOWNLOAD \
        $SKIP_MFA \
        $SKIP_EXTRACTION

log "All analyses complete."
