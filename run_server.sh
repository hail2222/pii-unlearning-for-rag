#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Server experiment runner
# Usage:
#   bash run_server.sh                        # Llama-3.1-8B (default)
#   bash run_server.sh --model Qwen/Qwen2.5-7B-Instruct
#   bash run_server.sh --resume               # resume interrupted run
# ─────────────────────────────────────────────────────────────

set -e

# ── Default settings ──────────────────────────────────────────
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DEVICE="cuda"
MAX_SAMPLES=200
PROBE_LAYER=24        # Llama-3.1-8B: 32 layers → upper-middle = 24
                      # Qwen2.5-7B  : 28 layers → upper-middle = 21
RESULTS_DIR="results_llama8b"
RESUME=""

# ── Parse optional overrides ──────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";       shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --probe-layer) PROBE_LAYER="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --resume)      RESUME="--resume"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Auto-set probe layer for known models ─────────────────────
if [[ "$MODEL" == *"Qwen2.5-7B"* ]]; then
    PROBE_LAYER=21
    RESULTS_DIR="results_qwen7b"
fi

echo "=============================================="
echo "Model      : $MODEL"
echo "Device     : $DEVICE"
echo "Max samples: $MAX_SAMPLES"
echo "Probe layer: $PROBE_LAYER"
echo "Results dir: $RESULTS_DIR"
echo "Resume     : ${RESUME:-no}"
echo "=============================================="

python run_experiment.py \
    --model       "$MODEL"       \
    --device      "$DEVICE"      \
    --max-samples "$MAX_SAMPLES" \
    --probe-layer "$PROBE_LAYER" \
    --results-dir "$RESULTS_DIR" \
    $RESUME

echo ""
echo "Experiment done. Running analysis..."
python analysis.py --results-dir "$RESULTS_DIR"

echo ""
echo "Running linear probe..."
python linear_probe.py --results-dir "$RESULTS_DIR"

echo ""
echo "Running end-to-end pipeline evaluation..."
python evaluate_pipeline.py --results-dir "$RESULTS_DIR"
