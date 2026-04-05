#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"

RESULTS_DIR="${1:-results/exp1_single_model}"
OUTPUT_DIR="${2:-results/thesis_visualizations/exp1_single_model}"

echo "[generate_exp1_thesis_artifacts] repo_root=$REPO_ROOT"
echo "[generate_exp1_thesis_artifacts] results_dir=$RESULTS_DIR"
echo "[generate_exp1_thesis_artifacts] output_dir=$OUTPUT_DIR"

python -m src.visualization.summary \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR"

python -m src.visualization.tables \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR"

python -m src.visualization.plots \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "[generate_exp1_thesis_artifacts] done"
