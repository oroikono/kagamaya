#!/bin/bash
#SBATCH --job-name=pdb5-preflight
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err
set -euo pipefail

ROOT="${PDEBENCH_RUN_ROOT:?set PDEBENCH_RUN_ROOT}"
SOURCE="${PDEBENCH_SOURCE_ROOT:?set PDEBENCH_SOURCE_ROOT}"
cd "$SOURCE"
python3 -m pdebench_diffreact_v5_20260828.preflight \
  --data "$ROOT/data/2D_diff-react_NA_NA.h5" \
  --output "$ROOT/results/v5_source_preflight.json"
