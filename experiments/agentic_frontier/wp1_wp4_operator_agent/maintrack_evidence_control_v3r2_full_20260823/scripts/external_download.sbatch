#!/bin/bash
#SBATCH --job-name=evctrl_v3r2_data
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/external_download_%j.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2/logs/external_download_%j.err
set -euo pipefail
ROOT=/cluster/scratch/ooikonomou/verisigs_evidence_control_v3r2
REPO="$ROOT/repo"
PY=/cluster/scratch/ooikonomou/verisigs_external_weakpde_v1/venv/bin/python
cd "$REPO"
PYTHONPATH="$REPO" "$PY" -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_evidence_control_v3r2_full_20260823.external_download \
  --manifest experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_evidence_control_v3r2_full_20260823/EXTERNAL_NUMERICAL_MANIFEST.json \
  --output-dir "$ROOT/external_data"
