#!/bin/bash
#SBATCH --job-name=pdb-download
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
set -euo pipefail

ROOT="${PDEBENCH_RUN_ROOT:?set PDEBENCH_RUN_ROOT}"
mkdir -p "$ROOT/data" "$ROOT/logs" "$ROOT/results"
TARGET="$ROOT/data/2D_diff-react_NA_NA.h5"
if [ ! -f "$TARGET" ]; then
  curl -L --fail --retry 5 --continue-at - \
    https://darus.uni-stuttgart.de/api/access/datafile/133017 -o "$TARGET"
fi
test "$(stat -c %s "$TARGET")" = "13243560688"
echo "b8d0b86064193195ddc30c33be5dc949  $TARGET" | md5sum --check --status
if [ ! -f "$ROOT/results/source.sha256" ]; then
  sha256sum "$TARGET" > "$ROOT/results/source.sha256"
fi
sha256sum --check --status "$ROOT/results/source.sha256"
printf 'PDEBENCH_SOURCE_READY %s\n' "$TARGET"
