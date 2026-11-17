#!/usr/bin/env bash
# Continue base into Impact, CombinedV2 into ImpactV2, or either validated
# CombinedV2 / ImpactV2 checkpoint into TASK=T800_base-ImpactV2Plus.
# Preview: CHECKPOINT=/absolute/path/ckpt_26500.pt bash scripts/train_impact.slurm --dry-run
# Submission is explicit: mkdir -p logs/slurm; sbatch --export=ALL,CHECKPOINT=... scripts/train_impact.slurm
#SBATCH --job-name=T800_base-Impact
#SBATCH --partition=rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/srv/shared/home/xjh/workspace/zq_leo_final
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

dry_run=false
if [[ $# -eq 1 && $1 == --dry-run ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    printf 'Usage: CHECKPOINT=... bash %s [--dry-run]\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Slurm runs a spooled script; --chdir supplies the checkout directory.
    PROJECT_ROOT="${T800_PROJECT_ROOT:-$PWD}"
else
    PROJECT_ROOT="${T800_PROJECT_ROOT:-$(dirname -- "$SCRIPT_DIR")}"
fi
PYTHON_BIN="${T800_PYTHON:-/srv/shared/home/xjh/app/miniconda3/envs/leo/bin/python}"
GPU_COUNT="${T800_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
NUM_ENVS="${NUM_ENVS:-4096}"
TASK="${TASK:-T800_base-Impact}"
if [[ -z "${CHECKPOINT:-}" ]]; then
    case "$TASK" in
        T800_base-ImpactV2-From154500-DR)
            CHECKPOINT="$PROJECT_ROOT/logs/cusrl/T800_base-ImpactV2/2026-11-09-19-36-41:T800_base-ImpactV2_2gpu_seed42/ckpt/ckpt_154500.pt"
            ;;
        T800_base-ImpactV2Plus-From173000-DR)
            CHECKPOINT="$PROJECT_ROOT/logs/cusrl/T800_base-ImpactV2Plus/2026-11-12-16-15-10:ImpactV2Plus_from_origin140000_2gpu_seed42/ckpt/ckpt_173000.pt"
            ;;
    esac
fi
SEED="${SEED:-42}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RUN_NAME="${RUN_NAME:-${TASK}_${GPU_COUNT}gpu_seed${SEED}}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
ADDITIONAL_ITERATIONS="${ADDITIONAL_ITERATIONS:-}"
IMPACT_SINGLE_HIT_MIGRATION="${IMPACT_SINGLE_HIT_MIGRATION:-0}"

if [[ "$TASK" != T800_base-Impact && "$TASK" != T800_base-ImpactV2 && "$TASK" != T800_base-ImpactV2Plus && "$TASK" != T800_base-ImpactV2-From154500-DR && "$TASK" != T800_base-ImpactV2Plus-From173000-DR ]]; then
    printf 'TASK must be a registered T800 Impact continuation task.\n' >&2
    exit 2
fi
if [[ -z "${CHECKPOINT:-}" ]]; then
    printf 'CHECKPOINT is required for %s; supply its validated continuation source.\n' "$TASK" >&2
    exit 2
fi
if [[ -n "$MAX_ITERATIONS" && -n "$ADDITIONAL_ITERATIONS" ]]; then
    printf 'MAX_ITERATIONS and ADDITIONAL_ITERATIONS are mutually exclusive.\n' >&2
    exit 2
fi
if [[ -n "$MAX_ITERATIONS" ]]; then
    BUDGET_NAME=MAX_ITERATIONS
    BUDGET_ARGS=(--max_iterations "$MAX_ITERATIONS")
else
    ADDITIONAL_ITERATIONS="${ADDITIONAL_ITERATIONS:-40000}"
    BUDGET_NAME=ADDITIONAL_ITERATIONS
    BUDGET_ARGS=(--additional_iterations "$ADDITIONAL_ITERATIONS")
fi
for variable in GPU_COUNT NUM_ENVS SAVE_INTERVAL "$BUDGET_NAME"; do
    if [[ ! "${!variable}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive decimal integer, without leading zeros.\n' "$variable" >&2
        exit 2
    fi
done
if [[ ! "$SEED" =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf 'SEED must be a nonnegative decimal integer, without leading zeros.\n' >&2
    exit 2
fi
if [[ "$IMPACT_SINGLE_HIT_MIGRATION" != 0 && "$IMPACT_SINGLE_HIT_MIGRATION" != 1 ]]; then
    printf 'IMPACT_SINGLE_HIT_MIGRATION must be 0 or 1.\n' >&2
    exit 2
fi
if [[ "$TASK" != T800_base-Impact && "$IMPACT_SINGLE_HIT_MIGRATION" != 0 ]]; then
    printf 'IMPACT_SINGLE_HIT_MIGRATION applies only to T800_base-Impact.\n' >&2
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
if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="$PWD/$CHECKPOINT"
fi
cd -- "$PROJECT_ROOT"
if [[ ! -f scripts/cusrl/train.py || ! -f scripts/check_config.py ]]; then
    printf 'T800_PROJECT_ROOT does not contain the training entry points: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi
if [[ ! -r "$CHECKPOINT" || ( ! -f "$CHECKPOINT" && ! -d "$CHECKPOINT" ) ]]; then
    printf 'CHECKPOINT must be a readable checkpoint file or run directory: %s\n' "$CHECKPOINT" >&2
    exit 1
fi

TRAIN_ARGS=(
    --task "$TASK" --headless --no-debug-vis
    --num_envs "$NUM_ENVS" --seed "$SEED" "${BUDGET_ARGS[@]}" --save_interval "$SAVE_INTERVAL"
    --logger tensorboard --log_dir logs/cusrl --run_name "$RUN_NAME" --checkpoint "$CHECKPOINT"
)
SMOKE_COMMAND=(
    "$PYTHON_BIN" -u scripts/check_config.py --task "$TASK" --checkpoint "$CHECKPOINT"
    --headless --device cuda:0 --num_envs 2 --smoke
)
if [[ "$IMPACT_SINGLE_HIT_MIGRATION" == 1 ]]; then
    TRAIN_ARGS+=(--impact-single-hit-migration)
    SMOKE_COMMAND+=(--impact-single-hit-migration)
fi
COMMAND=(
    "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nnodes=1
    "--nproc_per_node=$GPU_COUNT" scripts/cusrl/train.py "${TRAIN_ARGS[@]}"
)

printf 'Project: %s\nPython: %s\nGPUs: %s; environments: %s global / %s per GPU\n' \
    "$PWD" "$PYTHON_BIN" "$GPU_COUNT" "$NUM_ENVS" "$((NUM_ENVS / GPU_COUNT))"
printf 'Checkpoint: %s\n' "$CHECKPOINT"
printf 'Impact smoke check: '; printf '%q ' "${SMOKE_COMMAND[@]}"; printf '\n'
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
