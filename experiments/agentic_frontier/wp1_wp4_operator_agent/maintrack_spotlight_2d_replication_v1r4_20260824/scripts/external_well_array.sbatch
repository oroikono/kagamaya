#!/bin/bash
#SBATCH --job-name=spot-well
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=23:00:00
#SBATCH --output=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r4/logs/well_%x_%A_%a.out
#SBATCH --error=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r4/logs/well_%x_%A_%a.err

set -euo pipefail
umask 077

REPO_ROOT="${REPO_ROOT:?set REPO_ROOT to the source-frozen repository}"
RUN_ROOT="${RUN_ROOT:-/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r4}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the manifest-shaped The Well root}"
PYTHON_BIN="${PYTHON_BIN:-/cluster/home/ooikonomou/miniconda3/bin/python}"
SPLIT="${SPLIT:?set SPLIT to valid or test}"
SOURCE_CLOSURE="${SOURCE_CLOSURE:?set SOURCE_CLOSURE}"
AUTHORIZATION="${AUTHORIZATION:?set AUTHORIZATION path}"
INTERNAL_CONFIRMATION_AGGREGATE_SHA256="${INTERNAL_CONFIRMATION_AGGREGATE_SHA256:?set internal aggregate digest}"
EXTERNAL_VALIDATION_AGGREGATE_SHA256="${EXTERNAL_VALIDATION_AGGREGATE_SHA256:-}"
LANE_MODULE="experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r4_20260824"
MANIFEST="${REPO_ROOT}/experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r4_20260824/EXTERNAL_WELL_MANIFEST.json"
REGIMES=(bubbles gliders maze spirals spots worms)
REGIME="${REGIMES[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_ROOT="${RUN_ROOT}/results/external_${SPLIT}"

mkdir -p "${RUN_ROOT}/logs" "${OUTPUT_ROOT}" "${RUN_ROOT}/pycache/${SLURM_ARRAY_TASK_ID}"
cd "${REPO_ROOT}"
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/pycache/${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON_BIN}" -c 'import h5py,numpy; print({"h5py":h5py.__version__,"numpy":numpy.__version__})'

ARGS=(
  --split "${SPLIT}"
  --regime "${REGIME}"
  --data-root "${DATA_ROOT}"
  --manifest "${MANIFEST}"
  --source-closure "${SOURCE_CLOSURE}"
  --authorization "${AUTHORIZATION}"
  --internal-confirmation-aggregate-sha256 "${INTERNAL_CONFIRMATION_AGGREGATE_SHA256}"
  --output "${OUTPUT_ROOT}/external_${SPLIT}_${REGIME}.json.gz"
)
if [[ "${SPLIT}" == "test" ]]; then
  if [[ -z "${EXTERNAL_VALIDATION_AGGREGATE_SHA256}" ]]; then
    echo "TEST requires EXTERNAL_VALIDATION_AGGREGATE_SHA256" >&2
    exit 2
  fi
  ARGS+=(--external-validation-aggregate-sha256 "${EXTERNAL_VALIDATION_AGGREGATE_SHA256}")
fi

"${PYTHON_BIN}" -u -m "${LANE_MODULE}.external_well" "${ARGS[@]}"
"${PYTHON_BIN}" -m "${LANE_MODULE}.validate_external" \
  "${OUTPUT_ROOT}/external_${SPLIT}_${REGIME}.json.gz" \
  --manifest "${MANIFEST}" \
  --report "${OUTPUT_ROOT}/external_${SPLIT}_${REGIME}.validation.json"
sha256sum \
  "${OUTPUT_ROOT}/external_${SPLIT}_${REGIME}.json.gz" \
  "${OUTPUT_ROOT}/external_${SPLIT}_${REGIME}.validation.json" \
  > "${OUTPUT_ROOT}/external_${SPLIT}_${REGIME}.sha256"
