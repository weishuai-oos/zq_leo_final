#!/usr/bin/env bash
# Finite dynamic command evaluation; leaves running training unchanged.
#SBATCH --job-name=T800_agility_eval
#SBATCH --partition=rtx3090-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/srv/shared/home/xjh/workspace/zq_leo_final
#SBATCH --output=logs/slurm/T800_agility_eval-%j.out
#SBATCH --error=logs/slurm/T800_agility_eval-%j.err

set -euo pipefail
PYTHON_BIN="${T800_PYTHON:-/srv/shared/home/xjh/app/miniconda3/envs/leo/bin/python}"
: "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to a native stage-one checkpoint}"
: "${IMPACT_CHECKPOINT:?Set IMPACT_CHECKPOINT to an Impact checkpoint}"
EVAL_OUTPUT="${EVAL_OUTPUT:-docs/validation/agility_2026-11-04}"
export PYTHONUNBUFFERED=1
# Use the explicit environment Python in Slurm steps, as validated by the
# initialization probe. The batch-shell/OMP=1 path stalled before sim startup.
unset OMP_NUM_THREADS
mkdir -p "$EVAL_OUTPUT"

run_case() {
    local policy="$1" mode="$2" seed="$3" device="$4" checkpoint output
    local native_args=()
    if [[ "$policy" == stage1 ]]; then
        checkpoint="$BASE_CHECKPOINT"
        native_args=(--native-baseline)
    else
        checkpoint="$IMPACT_CHECKPOINT"
    fi
    output="$EVAL_OUTPUT/${policy}_${mode}_seed${seed}.json"
    printf 'Evaluating %s/%s seed=%s on %s\n' "$policy" "$mode" "$seed" "$device"
    # Each submitted attempt gets fresh artifacts, so Kit exit cannot reuse stale success.
    rm -f "$output" "${output%.json}.npz"
    srun --overlap --nodes=1 --ntasks=1 "$PYTHON_BIN" -u scripts/cusrl/evaluate_agility.py --checkpoint "$checkpoint" \
        --output "$output" --label "$policy" "${native_args[@]}" --num_envs 64 \
        --seed "$seed" --impact_mode "$mode" --headless --device "$device" \
        >"${output%.json}.out" 2>"${output%.json}.err"
    test -s "$output"
    test -s "${output%.json}.npz"
}

# First prove both restoration paths and dynamic timing before the full sweep.
run_case stage1 off 42 cuda:0 &
baseline_pid=$!
run_case impact off 42 cuda:1 &
impact_pid=$!
wait "$baseline_pid"
wait "$impact_pid"

run_case stage1 common 42 cuda:0 &
pids=("$!")
run_case impact common 42 cuda:1 &
pids+=("$!")
run_case stage1 off 43 cuda:2 &
pids+=("$!")
run_case impact off 43 cuda:3 &
pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

run_case stage1 common 43 cuda:0 &
baseline_pid=$!
run_case impact common 43 cuda:1 &
impact_pid=$!
wait "$baseline_pid"
wait "$impact_pid"
