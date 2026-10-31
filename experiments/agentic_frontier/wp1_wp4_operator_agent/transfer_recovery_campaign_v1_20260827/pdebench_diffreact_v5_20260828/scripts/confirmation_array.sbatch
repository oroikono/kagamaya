#!/bin/bash
#SBATCH --job-name=pdb5-confirm
#SBATCH --array=0-59
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/confirmation_%A_%a.out
#SBATCH --error=logs/confirmation_%A_%a.err
set -euo pipefail

ROOT="${PDEBENCH_RUN_ROOT:?set PDEBENCH_RUN_ROOT}"
SOURCE="${PDEBENCH_SOURCE_ROOT:?set PDEBENCH_SOURCE_ROOT}"
TASK="${SLURM_ARRAY_TASK_ID:?missing array task}"
OUTPUT=$(printf '%s/results/v5_confirmation_shards/confirmation_%03d.json' "$ROOT" "$TASK")
cd "$SOURCE"
python3 -m pdebench_diffreact_v5_20260828.confirmation \
  --data "$ROOT/data/2D_diff-react_NA_NA.h5" \
  --task-index "$TASK" \
  --source-receipt "$ROOT/results/v5_source_preflight.json" \
  --output "$OUTPUT"
