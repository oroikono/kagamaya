#!/bin/bash
#SBATCH --job-name=evctrl_v3r2_tc
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=4G
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/%A_%a.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/%A_%a.err

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SPLIT SOURCE_CLOSURE TASK_COUNT" >&2
  exit 2
fi
SPLIT="$1"
SOURCE_CLOSURE="$2"
TASK_COUNT="$3"
ROOT=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2
REPO="$ROOT/repo"
PY=/cluster/scratch/ooikonomou/verisigs_external_weakpde_v1/venv/bin/python

if [[ "$SLURM_ARRAY_TASK_COUNT" -ne "$TASK_COUNT" ]]; then
  echo "array cardinality mismatch" >&2
  exit 3
fi
case "$SPLIT" in
  train) BASE=61000 ;;
  calibration) BASE=62000 ;;
  *) echo "invalid split" >&2; exit 4 ;;
esac
SEED=$((BASE + SLURM_ARRAY_TASK_ID / 2))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then ORIGIN=spectral; else ORIGIN=fd; fi
OUT="$ROOT/results/$SPLIT/shard_${SEED}_${ORIGIN}.json.gz"
test ! -e "$OUT"
cd "$REPO"
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.freeze --verify
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.campaign \
  --split "$SPLIT" --seed "$SEED" --origin "$ORIGIN" \
  --source-closure "$SOURCE_CLOSURE" --output "$OUT"
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.validate "$OUT" \
  --report "$ROOT/results/$SPLIT/shard_${SEED}_${ORIGIN}.validation.json"
