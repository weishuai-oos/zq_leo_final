#!/usr/bin/env bash
# T800_base-AnkleCombined: train the combined ankle-stability task from scratch.
# Preview: bash scripts/train_ankle_combined.slurm --dry-run
# Submit:  mkdir -p logs/slurm && sbatch scripts/train_ankle_combined.slurm
# This launcher intentionally rejects CHECKPOINT so it cannot become an accidental resume.
#SBATCH --job-name=T800_ankle_combined
#SBATCH --partition=rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/srv/shared/home/xjh/workspace/zq_leo_final
#SBATCH --output=logs/slurm/T800_base-AnkleCombined-%j.out
#SBATCH --error=logs/slurm/T800_base-AnkleCombined-%j.err

set -euo pipefail

dry_run=false
if [[ $# -eq 1 && $1 == --dry-run ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
    exit 2
fi

if [[ -n "${CHECKPOINT:-}" ]]; then
    printf 'CHECKPOINT is forbidden for T800_base-AnkleCombined: this task must start from scratch.\n' >&2
    exit 2
fi

PROJECT_ROOT="${T800_PROJECT_ROOT:-/srv/shared/home/xjh/workspace/zq_leo_final}"
PYTHON_BIN="${T800_PYTHON:-/srv/shared/home/xjh/app/miniconda3/envs/leo/bin/python}"
GPU_COUNT="${T800_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-T800_base-AnkleCombined_${GPU_COUNT}gpu_seed${SEED}}"

for variable in GPU_COUNT NUM_ENVS MAX_ITERATIONS SAVE_INTERVAL; do
    if [[ ! "${!variable}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive decimal integer, without leading zeros.\n' "$variable" >&2
        exit 2
    fi
done
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'SEED must be a nonnegative decimal integer, without leading zeros.\n' "$SEED" >&2
    exit 2
fi
if (( NUM_ENVS % GPU_COUNT != 0 )); then
    printf 'NUM_ENVS=%s must be divisible by GPU_COUNT=%s.\n' "$NUM_ENVS" "$GPU_COUNT" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Leo Python is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$PROJECT_ROOT" || ! -r "$PROJECT_ROOT" ]]; then
    printf 'T800_PROJECT_ROOT must be an existing readable directory: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi

cd -- "$PROJECT_ROOT"
if [[ ! -f scripts/cusrl/train.py || ! -f scripts/check_config.py ]]; then
    printf 'T800_PROJECT_ROOT does not contain the training entry points: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi

TRAIN_ARGS=(
    --task T800_base-AnkleCombined --headless --no-debug-vis
    --num_envs "$NUM_ENVS" --seed "$SEED"
    --max_iterations "$MAX_ITERATIONS" --save_interval "$SAVE_INTERVAL"
    --logger tensorboard --log_dir logs/cusrl --run_name "$RUN_NAME"
)
COMMAND=(
    "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nnodes=1
    "--nproc_per_node=$GPU_COUNT" scripts/cusrl/train.py "${TRAIN_ARGS[@]}"
)
SMOKE_COMMAND=(
    "$PYTHON_BIN" -u scripts/check_config.py
    --task T800_base-AnkleCombined --headless --device cuda:0 --num-envs 2 --smoke
)

printf 'Project: %s\nPython: %s\nTask: T800_base-AnkleCombined\n' "$PROJECT_ROOT" "$PYTHON_BIN"
printf 'GPUs: %s; environments: %s global / %s per GPU\n' "$GPU_COUNT" "$NUM_ENVS" "$((NUM_ENVS / GPU_COUNT))"
printf 'Checkpoint: none (from scratch)\n'
printf 'GPU smoke check: '; printf '%q ' "${SMOKE_COMMAND[@]}"; printf '\n'
printf 'Training command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if "$dry_run"; then
    exit 0
fi

ENV_PREFIX="$(dirname -- "$(dirname -- "$PYTHON_BIN")")"
CONDA_SH=/srv/shared/home/xjh/app/miniconda3/etc/profile.d/conda.sh
if [[ -f "$CONDA_SH" && -d "$ENV_PREFIX/conda-meta" ]]; then
    source "$CONDA_SH"
    conda activate "$ENV_PREFIX"
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"$PYTHON_BIN" - "$GPU_COUNT" <<'PY'
import sys
import torch

requested = int(sys.argv[1])
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, got {sys.version}")
if not torch.cuda.is_available() or torch.cuda.device_count() < requested:
    raise SystemExit(f"Need {requested} usable GPUs; PyTorch sees {torch.cuda.device_count()}")
print(f"Torch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
for index in range(requested):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)
PY

"${SMOKE_COMMAND[@]}"
exec "${COMMAND[@]}"
