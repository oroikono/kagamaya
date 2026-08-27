#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 REPOSITORY_ROOT OUTPUT_TAR_GZ" >&2
  exit 2
fi
REPO="$1"
OUT="$2"
cd "$REPO"
PYTHONPATH="$REPO" .venv/bin/python -m \
  experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r7_20260824.scripts.stage_scientific \
  "$REPO" "$OUT"
