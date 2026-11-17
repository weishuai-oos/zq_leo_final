#!/usr/bin/env bash
# Final-checkpoint validation only; no training state is changed.
#SBATCH --job-name=T800_impact_final_eval
#SBATCH --partition=rtx4000
#SBATCH --nodelist=tr1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx4000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --chdir=/srv/shared/home/xjh/workspace/zq_leo_final
#SBATCH --output=logs/slurm/T800_impact_final_eval-%j.out
#SBATCH --error=logs/slurm/T800_impact_final_eval-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
unset OMP_NUM_THREADS
PYTHON_BIN=/srv/shared/home/xjh/app/miniconda3/envs/leo/bin/python
CHECKPOINT=logs/cusrl/T800_base-Impact/2026-11-04-20-26-01:T800_base-Impact_singlehit_groundfix_from115500_to125500/ckpt/ckpt_125500.pt
OUTPUT=docs/validation/impact_final_2026-11-05
mkdir -p "$OUTPUT"

run_case() {
    local mode="$1" seed="$2" device="$3" output="$OUTPUT/impact_${1}_seed${2}.json"
    rm -f "$output"
    srun --overlap --nodes=1 --ntasks=1 "$PYTHON_BIN" -u scripts/cusrl/evaluate_impact.py \
        --checkpoint "$CHECKPOINT" --output "$output" --label impact125500 \
        --num_envs 48 --duration-s 20 --seed "$seed" --impact_mode "$mode" \
        --headless --device "$device" >"${output%.json}.out" 2>"${output%.json}.err"
    test -s "$output"
}

for mode in off common stress; do
    run_case "$mode" 42 cuda:0 &
    first_pid=$!
    run_case "$mode" 43 cuda:1 &
    second_pid=$!
    wait "$first_pid"
    wait "$second_pid"
done

run_agility() {
    local seed="$1" device="$2" output="$OUTPUT/agility_off_seed${1}.json"
    rm -f "$output" "${output%.json}.npz"
    srun --overlap --nodes=1 --ntasks=1 "$PYTHON_BIN" -u scripts/cusrl/evaluate_agility.py \
        --checkpoint "$CHECKPOINT" --output "$output" --label impact125500 \
        --num_envs 64 --seed "$seed" --impact_mode off --headless --device "$device" \
        >"${output%.json}.out" 2>"${output%.json}.err"
    test -s "$output"
    test -s "${output%.json}.npz"
}

run_agility 42 cuda:0 &
first_pid=$!
run_agility 43 cuda:1 &
second_pid=$!
wait "$first_pid"
wait "$second_pid"
