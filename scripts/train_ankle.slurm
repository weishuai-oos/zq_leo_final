#!/usr/bin/env bash
# Four controlled ankle fine-tuning runs from the same 85500 checkpoint.
# Preview all commands: bash scripts/train_ankle.slurm --dry-run
# Submit all four: bash scripts/submit_ankle.sh
#SBATCH --job-name=T800_ankle
#SBATCH --partition=rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/srv/shared/home/xjh/workspace/zq_leo_final
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/ankle-%A_%a.out
#SBATCH --error=logs/slurm/ankle-%A_%a.err

set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi
DRY_RUN=false
[[ ${1:-} == --dry-run ]] && DRY_RUN=true

PROJECT_ROOT="${T800_PROJECT_ROOT:-/srv/shared/home/xjh/workspace/zq_leo_final}"
PYTHON_BIN="${T800_PYTHON:-/srv/shared/home/xjh/app/miniconda3/envs/leo/bin/python}"
GPU_COUNT="${T800_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-42}"
ADDITIONAL_ITERATIONS="${ADDITIONAL_ITERATIONS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
FINETUNE_LR_SCALE="${FINETUNE_LR_SCALE:-0.5}"
FINETUNE_WARMUP_ITERATIONS="${FINETUNE_WARMUP_ITERATIONS:-2000}"
CHECKPOINT="${CHECKPOINT:-/srv/shared/home/xjh/workspace/zq_leo_final/logs/cusrl/T800_base/2026-10-28-15-23-25:T800_base_2gpu_seed42/ckpt/ckpt_85500.pt}"

TASKS=(T800_base-AnkleSmooth T800_base-AnkleDamping T800_base-AnkleTargetLimit T800_base-AnklePose)
VARIANTS=(AnkleSmooth AnkleDamping AnkleTargetLimit AnklePose)

for variable in GPU_COUNT NUM_ENVS ADDITIONAL_ITERATIONS SAVE_INTERVAL FINETUNE_WARMUP_ITERATIONS; do
    if [[ ! "${!variable}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer.\n' "$variable" >&2
        exit 2
    fi
done
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'SEED must be a nonnegative integer.\n' >&2
    exit 2
fi
if (( NUM_ENVS % GPU_COUNT != 0 )); then
    printf 'NUM_ENVS=%s must be divisible by GPU_COUNT=%s.\n' "$NUM_ENVS" "$GPU_COUNT" >&2
    exit 2
fi
if [[ ! "$FINETUNE_LR_SCALE" =~ ^0\.[0-9]+$|^1$ ]] || [[ "$FINETUNE_LR_SCALE" =~ ^0\.0+$ ]]; then
    printf 'FINETUNE_LR_SCALE must be in (0,1].\n' >&2
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
if [[ ! -r "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    printf 'CHECKPOINT must be an existing readable checkpoint file: %s\n' "$CHECKPOINT" >&2
    exit 1
fi

build_command() {
    local index="$1"
    local output_name="$2"
    local task="${TASKS[$index]}"
    local run_name="${task}_from85500_${GPU_COUNT}gpu_seed${SEED}"
    local final_iteration
    final_iteration="$((85500 + ADDITIONAL_ITERATIONS))"
    local -n output_command="$output_name"
    output_command=(
        "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nnodes=1
        "--nproc_per_node=$GPU_COUNT" scripts/cusrl/train.py
        --task "$task" --headless --no-debug-vis
        --num_envs "$NUM_ENVS" --seed "$SEED"
        --max_iterations "$final_iteration" --save_interval "$SAVE_INTERVAL"
        --logger tensorboard --log_dir logs/cusrl --run_name "$run_name"
        --checkpoint "$CHECKPOINT"
        --finetune_lr_scale "$FINETUNE_LR_SCALE"
        --finetune_warmup_iterations "$FINETUNE_WARMUP_ITERATIONS"
        --finetune_origin_iteration 85500
    )
}

print_one() {
    local index="$1"
    local task="${TASKS[$index]}"
    local variant="${VARIANTS[$index]}"
    local run_name="${task}_from85500_${GPU_COUNT}gpu_seed${SEED}"
    local -a command=()
    build_command "$index" command
    printf 'Index %s: %s (%s)\n' "$index" "$task" "$variant"
    printf '  Checkpoint: %s\n' "$CHECKPOINT"
    printf '  Output: logs/cusrl/%s/<timestamp>:%s\n' "$task" "$run_name"
    printf '  Smoke: '; printf '%q ' "$PYTHON_BIN" -u scripts/check_config.py --task "$task" --checkpoint "$CHECKPOINT" --headless --device cuda:0 --num_envs 2 --smoke; printf '\n'
    printf '  Command: '; printf '%q ' "${command[@]}"; printf '\n'
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ ! "$SLURM_ARRAY_TASK_ID" =~ ^[0-3]$ ]]; then
        printf 'SLURM_ARRAY_TASK_ID must be 0..3.\n' >&2
        exit 2
    fi
    if "$DRY_RUN"; then
        print_one "$SLURM_ARRAY_TASK_ID"
        exit 0
    fi
    index="$SLURM_ARRAY_TASK_ID"
else
    if ! "$DRY_RUN"; then
        printf 'Run through sbatch array or set SLURM_ARRAY_TASK_ID; refusing an ambiguous direct launch.\n' >&2
        exit 2
    fi
    for index in 0 1 2 3; do print_one "$index"; done
    exit 0
fi

cd -- "$PROJECT_ROOT"
if [[ ! -f scripts/cusrl/train.py || ! -f scripts/check_config.py ]]; then
    printf 'T800_PROJECT_ROOT does not contain training entry points: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
ENV_PREFIX="$(dirname -- "$(dirname -- "$PYTHON_BIN")")"
CONDA_SH=/srv/shared/home/xjh/app/miniconda3/etc/profile.d/conda.sh
if [[ -f "$CONDA_SH" && -d "$ENV_PREFIX/conda-meta" ]]; then
    source "$CONDA_SH"
    conda activate "$ENV_PREFIX"
fi
"$PYTHON_BIN" - "$GPU_COUNT" <<'PY'
import sys, torch
requested = int(sys.argv[1])
if sys.version_info[:2] != (3, 11): raise SystemExit(f"Expected Python 3.11, got {sys.version}")
if not torch.cuda.is_available() or torch.cuda.device_count() < requested:
    raise SystemExit(f"Need {requested} usable GPUs; PyTorch sees {torch.cuda.device_count()}")
print(f"Torch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
PY

task="${TASKS[$index]}"
"$PYTHON_BIN" -u scripts/check_config.py --task "$task" --checkpoint "$CHECKPOINT" --headless --device cuda:0 --num_envs 2 --smoke
command=()
build_command "$index" command
exec "${command[@]}"
