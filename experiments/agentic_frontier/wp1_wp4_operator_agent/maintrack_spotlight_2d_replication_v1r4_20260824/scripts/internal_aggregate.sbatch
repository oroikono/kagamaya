#!/bin/bash
#SBATCH --job-name=spot2d_agg
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r4/logs/aggregate_%j.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r4/logs/aggregate_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ $# -ne 1 ]]; then echo "usage: $0 SPLIT" >&2; exit 2; fi
SPLIT="$1"
ROOT=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r4
REPO="$ROOT/repo"
PY=/cluster/scratch/ooikonomou/verisigs_external_weakpde_v1/venv/bin/python
LANE=experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r4_20260824
MODULE=experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r4_20260824
OUT="$ROOT/results/$SPLIT/aggregate.json"
test ! -e "$OUT"
cd "$REPO"
PYTHONPATH="$REPO" "$PY" -m "$MODULE.freeze" verify
PYTHONPATH="$REPO" "$PY" -m "$MODULE.aggregate" --split "$SPLIT" --input-dir "$ROOT/results/$SPLIT" --output "$OUT"
