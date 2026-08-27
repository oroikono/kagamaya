#!/bin/bash
set -euo pipefail

ROOT=/cluster/scratch/ooikonomou/verisigs_spotlight_2d_v1r4
REPO="$ROOT/repo"
LANE="$REPO/experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r4_20260824"
PY=/cluster/home/ooikonomou/miniconda3/bin/python
DATA_ROOT="$ROOT/data/the_well"
AUTH="$LANE/root_authorization_external_validation_76bb15c4063b0f76.json"
IC_SHA=929a69f0619eb5e9f5dc0993706988fb5ee262b3eb11ef1f02474eae627ec166
SOURCE_CLOSURE=07788f924f6e4ef2b053a9b30873965c72f992ac16ab803e25e5fc812cd4a583

test -s "$ROOT/results/confirmation/aggregate.json"
test -s "$AUTH"
test ! -e "$ROOT/results/external_valid_aggregate.json"
mkdir -p "$ROOT/logs" "$ROOT/receipts"

DL_JOB=$(sbatch --parsable \
  --job-name=spot-well-dl-valid \
  --time=08:00:00 \
  --cpus-per-task=1 \
  --mem-per-cpu=4G \
  --output="$ROOT/logs/well_download_valid_%j.out" \
  --error="$ROOT/logs/well_download_valid_%j.err" \
  --export=ALL,REPO_ROOT="$REPO",DATA_ROOT="$DATA_ROOT",PYTHON_BIN="$PY",AUTHORIZATION="$AUTH",INTERNAL_CONFIRMATION_AGGREGATE_SHA256="$IC_SHA",RECEIPT="$ROOT/receipts/external_valid_download.json" \
  --wrap='/bin/bash -lc '"'"'set -eu; module load eth_proxy; cd "$REPO_ROOT"; PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -u -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r4_20260824.scripts.download_external_well --split valid --regime all --manifest "$REPO_ROOT/experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r4_20260824/EXTERNAL_WELL_MANIFEST.json" --data-root "$DATA_ROOT" --authorization "$AUTHORIZATION" --internal-confirmation-aggregate-sha256 "$INTERNAL_CONFIRMATION_AGGREGATE_SHA256" --receipt "$RECEIPT"'"'"'')

VALID_ARRAY=$(sbatch --parsable \
  --dependency="afterok:$DL_JOB" \
  --export=ALL,REPO_ROOT="$REPO",RUN_ROOT="$ROOT",DATA_ROOT="$DATA_ROOT",PYTHON_BIN="$PY",SPLIT=valid,SOURCE_CLOSURE="$SOURCE_CLOSURE",AUTHORIZATION="$AUTH",INTERNAL_CONFIRMATION_AGGREGATE_SHA256="$IC_SHA" \
  "$LANE/scripts/external_well_array.sbatch")

VALID_AGG=$(sbatch --parsable \
  --dependency="afterok:$VALID_ARRAY" \
  --export=ALL,REPO_ROOT="$REPO",RUN_ROOT="$ROOT",PYTHON_BIN="$PY",SPLIT=valid \
  "$LANE/scripts/external_well_aggregate.sbatch")

printf '{"download_job":"%s","validation_array":"%s","validation_aggregate":"%s"}\n' \
  "$DL_JOB" "$VALID_ARRAY" "$VALID_AGG"
