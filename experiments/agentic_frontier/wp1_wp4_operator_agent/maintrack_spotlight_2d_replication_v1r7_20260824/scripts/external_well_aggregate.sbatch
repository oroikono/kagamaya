#!/bin/bash
#SBATCH --job-name=spot-well-agg
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r7/logs/well_aggregate_%j.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r7/logs/well_aggregate_%j.err

set -euo pipefail
umask 077

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the source-frozen repository}"
RUN_ROOT="${RUN_ROOT:-/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r7}"
PYTHON_BIN="${PYTHON_BIN:-/cluster/home/ooikonomou/miniconda3/bin/python}"
SPLIT="${SPLIT:?set SPLIT to valid or test}"
LANE_MODULE="experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r7_20260824"
MANIFEST="${REPO_ROOT}/experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r7_20260824/EXTERNAL_WELL_MANIFEST.json"
INPUT_ROOT="${RUN_ROOT}/results/external_${SPLIT}"
OUTPUT="${RUN_ROOT}/results/external_${SPLIT}_aggregate.json"

cd "${REPO_ROOT}"
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/pycache/aggregate_${SLURM_JOB_ID}"
"${PYTHON_BIN}" -m "${LANE_MODULE}.aggregate_external" \
  --split "${SPLIT}" \
  --input-dir "${INPUT_ROOT}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT}"
sha256sum "${OUTPUT}" > "${OUTPUT}.sha256"
