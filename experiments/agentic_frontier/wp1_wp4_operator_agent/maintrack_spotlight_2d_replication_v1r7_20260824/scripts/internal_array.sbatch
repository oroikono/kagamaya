#!/bin/bash
#SBATCH --job-name=spot2d_v1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=4G
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r7/logs/%A_%a.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r7/logs/%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 SPLIT SOURCE_CLOSURE AUTHORIZATION_BASENAME TASK_COUNT [UPSTREAM_AGGREGATE_SHA256]" >&2
  exit 2
fi
SPLIT="$1"
SOURCE_CLOSURE="$2"
AUTH_BASENAME="$3"
TASK_COUNT="$4"
UPSTREAM="${5:-}"
ROOT=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r7
REPO="$ROOT/repo"
PY=/cluster/scratch/ooikonomou/verisigs_external_weakpde_v1/venv/bin/python
LANE=experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r7_20260824

if [[ "$SLURM_ARRAY_TASK_COUNT" -ne "$TASK_COUNT" ]]; then
  echo "array cardinality mismatch" >&2
  exit 3
fi
case "$SPLIT" in
  mechanism) BASE=161000; EXPECTED=40 ;;
  confirmation) BASE=162000; EXPECTED=200 ;;
  *) echo "invalid split" >&2; exit 4 ;;
esac
if [[ "$TASK_COUNT" -ne "$EXPECTED" ]]; then
  echo "split/task-count mismatch" >&2
  exit 5
fi
SEED=$((BASE + SLURM_ARRAY_TASK_ID / 2))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then ORIGIN=spectral; else ORIGIN=fd; fi
OUT="$ROOT/results/$SPLIT/shard_${SEED}_${ORIGIN}.json.gz"
REPORT="$ROOT/results/$SPLIT/shard_${SEED}_${ORIGIN}.validation.json"
test ! -e "$OUT"
test ! -e "$REPORT"
cd "$REPO"
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r7_20260824.freeze verify
ARGS=(--split "$SPLIT" --seed "$SEED" --origin "$ORIGIN" --source-closure "$SOURCE_CLOSURE" --authorization "$LANE/$AUTH_BASENAME" --output "$OUT")
if [[ "$SPLIT" == confirmation ]]; then
  test -n "$UPSTREAM"
  ARGS+=(--upstream-aggregate-sha256 "$UPSTREAM")
fi
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r7_20260824.campaign "${ARGS[@]}"
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r7_20260824.validate "$OUT" --report "$REPORT"
