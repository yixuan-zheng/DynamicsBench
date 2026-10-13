#!/bin/bash
#SBATCH --job-name=dynamicsbench-evaluation
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-5%4
#SBATCH --output=results/pace/policy-evaluation/run-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda/12.1.1
module load pytorch/2.1.0
export APPTAINERENV_PYTHONPATH="$PWD/src"
export APPTAINERENV_OMP_NUM_THREADS=2
export APPTAINERENV_CUBLAS_WORKSPACE_CONFIG=:4096:8
alphas=(1 1 1 16 16 16)
seeds=(11 12 13 11 12 13)
case_dir="results/pace/policy-evaluation/alpha${alphas[$SLURM_ARRAY_TASK_ID]}-seed${seeds[$SLURM_ARRAY_TASK_ID]}"
mkdir -p "$case_dir"
python examples/policy_evaluation.py --alpha "${alphas[$SLURM_ARRAY_TASK_ID]}" --seed "${seeds[$SLURM_ARRAY_TASK_ID]}" --output "$case_dir"
python -m pip freeze > "$case_dir/environment-lock.txt"
