#!/usr/bin/env bash
# Continue Combined into AnkleCombinedV2; no job is submitted by this file itself.
# Preview: bash scripts/train_ankle_combined_v2.slurm --dry-run
# Submit from the project: mkdir -p logs/slurm && sbatch scripts/train_ankle_combined_v2.slurm
# Override CHECKPOINT_PATH to use another Combined source or resume matching V2.
# The Python checkpoint contract validates task, configuration and saved progress.
#SBATCH --job-name=T800_ankle_combined_v2
#SBATCH --partition=rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/srv/shared/home/xjh/workspace/zq_leo_final
#SBATCH --output=logs/slurm/T800_base-AnkleCombinedV2-%j.out
#SBATCH --error=logs/slurm/T800_base-AnkleCombinedV2-%j.err

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
ADDITIONAL_ITERATIONS="${ADDITIONAL_ITERATIONS:-40000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
FINETUNE_LR_SCALE="${FINETUNE_LR_SCALE:-0.5}"
FINETUNE_WARMUP_ITERATIONS="${FINETUNE_WARMUP_ITERATIONS:-1000}"
FINETUNE_ORIGIN_ITERATION="${FINETUNE_ORIGIN_ITERATION:-}"
RUN_NAME="${RUN_NAME:-T800_base-AnkleCombinedV2_finetune_${GPU_COUNT}gpu_seed${SEED}}"
DEFAULT_CHECKPOINT="$PROJECT_ROOT/logs/cusrl/T800_base-AnkleCombined/2026-11-04-00-15-03:T800_base-AnkleCombined_2gpu_seed42/ckpt/ckpt_100000.pt"

if [[ -n "${CHECKPOINT_PATH:-}" && -n "${CHECKPOINT:-}" && "$CHECKPOINT_PATH" != "$CHECKPOINT" ]]; then
    printf 'CHECKPOINT_PATH and CHECKPOINT refer to different files; set only one.\n' >&2
    exit 2
fi
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT:-$DEFAULT_CHECKPOINT}}"
if [[ -n "${MAX_ITERATIONS:-}" ]]; then
    printf 'Use ADDITIONAL_ITERATIONS for this continuation launcher, not MAX_ITERATIONS.\n' >&2
    exit 2
fi
for variable in GPU_COUNT NUM_ENVS ADDITIONAL_ITERATIONS SAVE_INTERVAL; do
    if [[ ! "${!variable}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive decimal integer, without leading zeros.\n' "$variable" >&2
        exit 2
    fi
done
for variable in SEED FINETUNE_WARMUP_ITERATIONS; do
    if [[ ! "${!variable}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        printf '%s must be a nonnegative decimal integer, without leading zeros.\n' "$variable" >&2
        exit 2
    fi
done
if [[ -n "$FINETUNE_ORIGIN_ITERATION" && ! "$FINETUNE_ORIGIN_ITERATION" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'FINETUNE_ORIGIN_ITERATION must be a nonnegative decimal integer.\n' >&2
    exit 2
fi
if [[ ! "$FINETUNE_LR_SCALE" =~ ^0\.[0-9]*[1-9][0-9]*$|^1(\.0+)?$ ]]; then
    printf 'FINETUNE_LR_SCALE must be in (0, 1].\n' >&2
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
# Interpret an explicitly supplied relative checkpoint against the caller's cwd.
if [[ "$CHECKPOINT_PATH" != /* ]]; then
    CHECKPOINT_PATH="$PWD/$CHECKPOINT_PATH"
fi
if [[ ! -f "$CHECKPOINT_PATH" || ! -r "$CHECKPOINT_PATH" ]]; then
    printf 'CHECKPOINT_PATH must be an existing readable checkpoint file: %s\n' "$CHECKPOINT_PATH" >&2
    exit 1
fi

cd -- "$PROJECT_ROOT"
if [[ ! -f scripts/cusrl/train.py || ! -f scripts/check_config.py ]]; then
    printf 'T800_PROJECT_ROOT does not contain the training entry points: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi

TRAIN_ARGS=(
    --task T800_base-AnkleCombinedV2 --headless --no-debug-vis
    --num_envs "$NUM_ENVS" --seed "$SEED"
    --additional_iterations "$ADDITIONAL_ITERATIONS" --save_interval "$SAVE_INTERVAL"
    --logger tensorboard --log_dir logs/cusrl --run_name "$RUN_NAME"
    --checkpoint "$CHECKPOINT_PATH"
    --finetune_lr_scale "$FINETUNE_LR_SCALE"
    --finetune_warmup_iterations "$FINETUNE_WARMUP_ITERATIONS"
)
if [[ -n "$FINETUNE_ORIGIN_ITERATION" ]]; then
    TRAIN_ARGS+=(--finetune_origin_iteration "$FINETUNE_ORIGIN_ITERATION")
fi
COMMAND=(
    "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nnodes=1 --max_restarts=0
    "--nproc_per_node=$GPU_COUNT" scripts/cusrl/train.py "${TRAIN_ARGS[@]}"
)
SMOKE_COMMAND=(
    "$PYTHON_BIN" -u scripts/check_config.py
    --task T800_base-AnkleCombinedV2 --checkpoint "$CHECKPOINT_PATH"
    --finetune-lr-scale "$FINETUNE_LR_SCALE"
    --finetune-warmup-iterations "$FINETUNE_WARMUP_ITERATIONS"
    --headless --device cuda:0 --num-envs 2 --smoke
)
if [[ -n "$FINETUNE_ORIGIN_ITERATION" ]]; then
    SMOKE_COMMAND+=(--finetune-origin-iteration "$FINETUNE_ORIGIN_ITERATION")
fi

printf 'Project: %s\nPython: %s\nTask: T800_base-AnkleCombinedV2\n' "$PROJECT_ROOT" "$PYTHON_BIN"
printf 'GPUs: %s; environments: %s global / %s per GPU\n' "$GPU_COUNT" "$NUM_ENVS" "$((NUM_ENVS / GPU_COUNT))"
printf 'Checkpoint: %s\nAdditional iterations: %s\n' "$CHECKPOINT_PATH" "$ADDITIONAL_ITERATIONS"
printf 'Output: logs/cusrl/T800_base-AnkleCombinedV2/<timestamp>:%s\n' "$RUN_NAME"
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

# Keep both stages in the foreground. A smoke/rank failure exits the batch job;
# torchrun owns rank cleanup, and exec leaves no waiting background shell behind.
"${SMOKE_COMMAND[@]}"
exec "${COMMAND[@]}"
