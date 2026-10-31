#!/bin/bash
#SBATCH --job-name=pdb-audit
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/audit_%j.out
#SBATCH --error=logs/audit_%j.err
set -euo pipefail

ROOT="${PDEBENCH_RUN_ROOT:?set PDEBENCH_RUN_ROOT}"
SOURCE="${PDEBENCH_SOURCE_ROOT:?set PDEBENCH_SOURCE_ROOT}"
cd "$SOURCE"
python3 -m pdebench_diffreact_v5_20260828.pipeline \
  --stage audit \
  --data "$ROOT/data/2D_diff-react_NA_NA.h5" \
  --threshold-file "$ROOT/results/development_thresholds.json" \
  --output "$ROOT/results/authority_audit.json"
