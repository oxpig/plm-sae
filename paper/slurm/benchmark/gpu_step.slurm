#!/bin/bash
#SBATCH --job-name=plm_step_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# One GPU step of the benchmark: python -m $STEP (module and arguments), with the environment of scripts/env.sh.
# Submit from the repository root, for example
#   sbatch --export=ALL,PLM_SAE_REPO="$PWD",STEP="paper.benchmark.positions piggen" paper/slurm/benchmark/gpu_step.slurm
#   sbatch --export=ALL,PLM_SAE_REPO="$PWD",STEP="plm_sae.steering.output_scores IgLM" paper/slurm/benchmark/gpu_step.slurm
#   sbatch --export=ALL,PLM_SAE_REPO="$PWD",STEP="paper.benchmark.natural_scale measure progen2_oas" paper/slurm/benchmark/gpu_step.slurm
# The steps are listed in paper/README.md (Steering benchmark).
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
[ -n "${STEP:-}" ] || { echo "set STEP"; exit 1; }
"$PY" -m $STEP
