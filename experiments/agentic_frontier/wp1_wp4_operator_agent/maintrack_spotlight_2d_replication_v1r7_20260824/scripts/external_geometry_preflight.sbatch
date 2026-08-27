#!/bin/bash
#SBATCH --job-name=spot-geometry-preflight
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r7/logs/geometry_preflight_%j.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r7/logs/geometry_preflight_%j.err

set -euo pipefail
umask 077

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the pre-freeze v1r7 source tree}"
RUN_ROOT="${RUN_ROOT:-/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r7}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the manifest-shaped The Well root}"
PYTHON_BIN="${PYTHON_BIN:-/cluster/home/ooikonomou/miniconda3/bin/python}"
PREFLIGHT_STAGE_ARCHIVE_SHA256="${PREFLIGHT_STAGE_ARCHIVE_SHA256:?set the digest printed by the deterministic preflight stager}"
LANE_MODULE="experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r7_20260824"
LANE_PATH="${REPO_ROOT}/experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r7_20260824"
UPSTREAM_V1R6_ROOT="${UPSTREAM_V1R6_ROOT:-${LANE_PATH}/upstream_v1r6}"
MANIFEST="${LANE_PATH}/EXTERNAL_WELL_MANIFEST.json"
RECEIPT="${RUN_ROOT}/receipts/EXTERNAL_GEOMETRY_PREFLIGHT_VALID.json"
INTERNAL_SHA="929a69f0619eb5e9f5dc0993706988fb5ee262b3eb11ef1f02474eae627ec166"
V1R6_AUTH="${UPSTREAM_V1R6_ROOT}/root_authorization_external_validation_8f91ff35c7e42c1f.json"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/receipts" "${RUN_ROOT}/pycache/preflight"
cd "${REPO_ROOT}"
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/pycache/preflight"

"${PYTHON_BIN}" -u -m "${LANE_MODULE}.scripts.preflight_external_geometry" \
  --manifest "${MANIFEST}" \
  --data-root "${DATA_ROOT}" \
  --upstream-v1r6-freeze "${UPSTREAM_V1R6_ROOT}/PREOUTCOME_FREEZE.json" \
  --upstream-v1r6-authorization "${V1R6_AUTH}" \
  --internal-confirmation-aggregate-sha256 "${INTERNAL_SHA}" \
  --preflight-stage-archive-sha256 "${PREFLIGHT_STAGE_ARCHIVE_SHA256}" \
  --receipt "${RECEIPT}"

sha256sum "${RECEIPT}"
