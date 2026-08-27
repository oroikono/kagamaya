#!/bin/bash
#SBATCH --job-name=evctrl_v3r2_ext
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=8G
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/external_eval_%j.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/external_eval_%j.err
set -euo pipefail
ROOT=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2
REPO="$ROOT/repo"
PY=/cluster/scratch/ooikonomou/verisigs_external_weakpde_v1/venv/bin/python
cd "$REPO"
PYDEPS="$ROOT/python_deps"
PYTHONPATH="$PYDEPS:$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.external_transfer \
  --data-root "$ROOT/external_data" \
  --manifest experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_evidence_control_v3r2_full_20260823/EXTERNAL_NUMERICAL_MANIFEST.json \
  --receipt "$ROOT/results/model_receipt/receipt.json" \
  --dominance "$ROOT/results/calibration_dominance.json" \
  --output "$ROOT/results/EXTERNAL_FHN_TRANSFER.json"
PYTHONPATH="$PYDEPS:$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.validate_external \
  "$ROOT/results/EXTERNAL_FHN_TRANSFER.json" \
  --manifest experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_evidence_control_v3r2_full_20260823/EXTERNAL_NUMERICAL_MANIFEST.json \
  --receipt "$ROOT/results/model_receipt/receipt.json" \
  --report "$ROOT/results/EXTERNAL_FHN_TRANSFER.validation.json"
