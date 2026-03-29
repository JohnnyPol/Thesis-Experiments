#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

WEIGHTS_PATH="${PROJECT_ROOT}/checkpoints/baseline/resnet18_baseline.pth"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/results/exp1_single_model/01_single_node_baseline/run_001"

python -m src.inference.single_node \
  --weights "$WEIGHTS_PATH" \
  --data-dir "$DATA_DIR" \
  --batch-size 1 \
  --device cpu \
  --output-dir "$OUTPUT_DIR" \
  --network-interface eth0 \
  --warmup-samples 20 \
  --model-name resnet18