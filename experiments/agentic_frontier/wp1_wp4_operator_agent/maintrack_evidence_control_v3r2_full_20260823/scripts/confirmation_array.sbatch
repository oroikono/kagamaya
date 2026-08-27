#!/bin/bash
#SBATCH --job-name=evctrl_v3r2_confirm
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=4G
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/%A_%a.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/%A_%a.err

set -euo pipefail

SOURCE_CLOSURE="$1"
AUTH="$2"
AUTH_SHA="$3"
ROOT=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2
REPO="$ROOT/repo"
PY=/cluster/scratch/ooikonomou/verisigs_external_weakpde_v1/venv/bin/python
test "$SLURM_ARRAY_TASK_COUNT" -eq 200
SEED=$((63000 + SLURM_ARRAY_TASK_ID / 2))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then ORIGIN=spectral; else ORIGIN=fd; fi
OUT="$ROOT/results/confirmation/shard_${SEED}_${ORIGIN}.json.gz"
test ! -e "$OUT"
cd "$REPO"
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.freeze --verify
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.campaign \
  --split confirmation --seed "$SEED" --origin "$ORIGIN" \
  --source-closure "$SOURCE_CLOSURE" --authorization "$AUTH" \
  --authorization-sha "$AUTH_SHA" --output "$OUT"
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.validate "$OUT" \
  --report "$ROOT/results/confirmation/shard_${SEED}_${ORIGIN}.validation.json"
