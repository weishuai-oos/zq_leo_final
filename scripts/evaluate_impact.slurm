#!/usr/bin/env bash
# Paired finite V2/Plus benchmark. Sources are explicit; nothing is submitted by this file.
#SBATCH --job-name=T800_impact_eval
#SBATCH --partition=rtx3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --chdir=/srv/shared/home/xjh/workspace/zq_leo_final
#SBATCH --output=logs/slurm/T800_impact_eval-%j.out
#SBATCH --error=logs/slurm/T800_impact_eval-%j.err

set -euo pipefail
PYTHON_BIN="${T800_PYTHON:-/srv/shared/home/xjh/app/miniconda3/envs/leo/bin/python}"
: "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to policy A, normally an ImpactV2 checkpoint}"
: "${IMPACT_CHECKPOINT:?Set IMPACT_CHECKPOINT to policy B, normally an ImpactV2Plus checkpoint}"
TASK_A="${TASK_A:-T800_base-ImpactV2-Play}"
TASK_B="${TASK_B:-T800_base-ImpactV2Plus-Play}"
NATIVE_BASELINE_A="${NATIVE_BASELINE_A:-0}"
PROTOCOL="${PROTOCOL:-recovery_v1}"
EVAL_OUTPUT="${EVAL_OUTPUT:-logs/evaluation/impact_v2_plus/${SLURM_JOB_ID:-manual}}"
EVAL_GPU_COUNT="${EVAL_GPU_COUNT:-1}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-48}"
EVAL_DURATION_S="${EVAL_DURATION_S:-20}"
if [[ ! "$EVAL_GPU_COUNT" =~ ^[1-9][0-9]*$ || ! "$EVAL_NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'EVAL_GPU_COUNT and EVAL_NUM_ENVS must be positive integers.\n' >&2
    exit 2
fi
if [[ "$NATIVE_BASELINE_A" != 0 && "$NATIVE_BASELINE_A" != 1 ]]; then
    printf 'NATIVE_BASELINE_A must be 0 or 1.\n' >&2
    exit 2
fi
source /srv/shared/home/xjh/app/miniconda3/etc/profile.d/conda.sh
conda activate "$(dirname -- "$(dirname -- "$PYTHON_BIN")")"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p "$EVAL_OUTPUT"

"$PYTHON_BIN" -u scripts/check_config.py --task "${TASK_B%-Play}" \
    --checkpoint "$IMPACT_CHECKPOINT" --headless --device cuda:0 --num_envs 2 --smoke

case_index=0
pids=()
for seed in 42 43; do
    for mode in off common stress; do
        for policy in a b; do
            native_args=()
            if [[ "$policy" == a ]]; then
                checkpoint="$BASE_CHECKPOINT"; task="$TASK_A"
                [[ "$NATIVE_BASELINE_A" == 0 ]] || native_args=(--native-baseline)
            else
                checkpoint="$IMPACT_CHECKPOINT"; task="$TASK_B"
            fi
            output="$EVAL_OUTPUT/${policy}_${mode}_seed${seed}.json"
            device="cuda:$((case_index % EVAL_GPU_COUNT))"
            printf 'Evaluating %s task=%s mode=%s seed=%s device=%s\n' "$policy" "$task" "$mode" "$seed" "$device"
            (
                "$PYTHON_BIN" -u scripts/cusrl/evaluate_impact.py --task "$task" --checkpoint "$checkpoint" \
                    --output "$output" --label "$policy" "${native_args[@]}" --protocol "$PROTOCOL" \
                    --num_envs "$EVAL_NUM_ENVS" --duration-s "$EVAL_DURATION_S" --seed "$seed" \
                    --impact_mode "$mode" --headless --device "$device"
                test -s "$output"
            ) &
            pids+=("$!")
            case_index=$((case_index + 1))
            if (( case_index % EVAL_GPU_COUNT == 0 )); then
                for pid in "${pids[@]}"; do wait "$pid"; done
                pids=()
            fi
        done
    done
done
for pid in "${pids[@]}"; do wait "$pid"; done
