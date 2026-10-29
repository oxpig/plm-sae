#!/bin/bash
#SBATCH --job-name=plm_step_cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=06:00:00
# One CPU step of the benchmark: python -m $STEP (module and arguments), with the environment of scripts/env.sh and
# NPROC = the job's CPUs. Submit from the repository root, for example
#   sbatch --export=ALL,PLM_SAE_REPO="$PWD",STEP="paper.benchmark.analyze_final" paper/slurm/benchmark/cpu_step.slurm
#   sbatch --export=ALL,PLM_SAE_REPO="$PWD",STEP="paper.benchmark.strict existing" paper/slurm/benchmark/cpu_step.slurm
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
[ -n "${STEP:-}" ] || { echo "set STEP"; exit 1; }
export NPROC=${NPROC:-${SLURM_CPUS_PER_TASK:-24}}
"$PY" -m $STEP
