#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 REPOSITORY_ROOT OUTPUT_TAR_GZ" >&2
  exit 2
fi
REPO="$1"
OUT="$2"
LANE=experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_spotlight_2d_replication_v1r4_20260824
cd "$REPO"
test -f "$LANE/PREOUTCOME_FREEZE.json"
PYTHONPATH="$REPO" .venv/bin/python -m experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_spotlight_2d_replication_v1r4_20260824.freeze verify
test ! -e "$OUT"
COPYFILE_DISABLE=1 tar -czf "$OUT" \
  experiments/agentic_frontier/wp1_wp4_operator_agent/__init__.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/config.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/engine.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/manufactured.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/proposals.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/run_demo.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/serialization.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/susano_candidates.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/validate_artifacts.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/weak_form.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/__init__.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/contract.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_2d_physical_v1/__init__.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_2d_physical_v1/exact_2d_oracle.py \
  experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_2d_physical_v1/exact_2d_coupled_oracle.py \
  "$LANE"
shasum -a 256 "$OUT"
