#!/bin/bash
#SBATCH --job-name=plm_ctl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
# Runs one orchestration loop of paper/scripts/benchmark (it only polls, submits numbering jobs and runs short
# evaluations), so that it does not run on a login node. The loops are idempotent (queued markers, existing child
# tasks): a requeued or resubmitted controller resumes. Submit from the repository root, for example
#   sbatch --export=ALL,PLM_SAE_REPO="$PWD",LOOP=chain paper/slurm/benchmark/controller.slurm
# LOOP = chain | windowed_number | progen2_t10_number | progen2_t10_ladder.
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
[ -n "${LOOP:-}" ] || { echo "set LOOP"; exit 1; }
bash "$PLM_SAE_REPO/paper/scripts/benchmark/$LOOP.sh"
