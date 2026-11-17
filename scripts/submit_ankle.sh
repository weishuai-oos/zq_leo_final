#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname -- "$SCRIPT_DIR")"
mkdir -p "$PROJECT_ROOT/logs/slurm"
if [[ ${1:-} == --dry-run ]]; then
    exec bash "$SCRIPT_DIR/train_ankle.slurm" --dry-run
fi
if [[ $# -ne 0 ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi
exec sbatch --chdir="$PROJECT_ROOT" "$SCRIPT_DIR/train_ankle.slurm"
