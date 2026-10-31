#!/bin/bash
#SBATCH --job-name=pdb5-aggregate
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err
set -euo pipefail

ROOT="${PDEBENCH_RUN_ROOT:?set PDEBENCH_RUN_ROOT}"
SOURCE="${PDEBENCH_SOURCE_ROOT:?set PDEBENCH_SOURCE_ROOT}"
cd "$SOURCE"
python3 -m pdebench_diffreact_v5_20260828.aggregate \
  --shard-dir "$ROOT/results/v5_confirmation_shards" \
  --output "$ROOT/results/v5_confirmation_aggregate.json"
