#!/usr/bin/env bash
# T800_base: train from scratch unless CHECKPOINT is explicitly set.
# 4096 global environments / 2 GPUs = 2048 environments per GPU.
# Preview: bash scripts/train.slurm --dry-run
# Before submitting on a fresh checkout: mkdir -p logs/slurm
# Submit:  sbatch scripts/train.slurm
#SBATCH --job-name=T800_base
#SBATCH --partition=rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/srv/shared/home/xjh/workspace/zq_leo_final
#SBATCH --output=logs/slurm/T800_base-%j.out
#SBATCH --error=logs/slurm/T800_base-%j.err

set -euo pipefail

dry_run=false
if [[ $# -eq 1 && $1 == --dry-run ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    printf 'Usage: bash %s [--dry-run]\n' "$0" >&2
    exit 2
fi

PROJECT_ROOT="${T800_PROJECT_ROOT:-/srv/shared/home/xjh/workspace/zq_leo_final}"
PYTHON_BIN="${T800_PYTHON:-/srv/shared/home/xjh/app/miniconda3/envs/leo/bin/python}"
GPU_COUNT="${T800_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-42}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-T800_base_${GPU_COUNT}gpu_seed${SEED}}"

for variable in GPU_COUNT NUM_ENVS MAX_ITERATIONS SAVE_INTERVAL; do
    if [[ ! "${!variable}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive decimal integer, without leading zeros.\n' "$variable" >&2
        exit 2
    fi
done
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'SEED must be a nonnegative decimal integer, without leading zeros.\n' >&2
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
cd -- "$PROJECT_ROOT"
if [[ ! -f scripts/cusrl/train.py || ! -f scripts/check_config.py ]]; then
    printf 'T800_PROJECT_ROOT does not contain the training entry points: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi

TRAIN_ARGS=(
    --task T800_base --headless --no-debug-vis
    --num_envs "$NUM_ENVS" --seed "$SEED"
    --max_iterations "$MAX_ITERATIONS" --save_interval "$SAVE_INTERVAL"
    --logger tensorboard --log_dir logs/cusrl --run_name "$RUN_NAME"
)
if [[ -n "${CHECKPOINT:-}" ]]; then
    if [[ ! -r "$CHECKPOINT" || ( ! -f "$CHECKPOINT" && ! -d "$CHECKPOINT" ) ]]; then
        printf 'CHECKPOINT must be a readable checkpoint file or run directory: %s\n' "$CHECKPOINT" >&2
        exit 1
    fi
    TRAIN_ARGS+=(--checkpoint "$CHECKPOINT")
fi
COMMAND=(
    "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nnodes=1
    "--nproc_per_node=$GPU_COUNT" scripts/cusrl/train.py "${TRAIN_ARGS[@]}"
)
SMOKE_COMMAND=("$PYTHON_BIN" -u scripts/check_config.py --headless --device cuda:0 --smoke)

printf 'Project: %s\nPython: %s\nGPUs: %s; environments: %s global / %s per GPU\n' \
    "$PWD" "$PYTHON_BIN" "$GPU_COUNT" "$NUM_ENVS" "$((NUM_ENVS / GPU_COUNT))"
printf 'Checkpoint: %s\n' "${CHECKPOINT:-none (from scratch)}"
printf 'GPU smoke check: '; printf '%q ' "${SMOKE_COMMAND[@]}"; printf '\n'
printf 'Training command: '; printf '%q ' "${COMMAND[@]}"; printf '\n'
if "$dry_run"; then
    exit 0
fi

# Activate the selected Conda prefix so Kit subprocesses see the same environment.
ENV_PREFIX="$(dirname -- "$(dirname -- "$PYTHON_BIN")")"
CONDA_SH=/srv/shared/home/xjh/app/miniconda3/etc/profile.d/conda.sh
if [[ -f "$CONDA_SH" && -d "$ENV_PREFIX/conda-meta" ]]; then
    source "$CONDA_SH"
    conda activate "$ENV_PREFIX"
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Keep Slurm's CUDA_VISIBLE_DEVICES. Do not expose GPUs assigned to other jobs.
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

# Abort before a long run if URDF conversion, reset, AMP sampling or stepping fails.
"${SMOKE_COMMAND[@]}"
exec "${COMMAND[@]}"
