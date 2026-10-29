#!/bin/bash
#SBATCH --job-name=plm_gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# One GPU worker of the steering pipelines (paper/scripts/benchmark/gpu_worker.sh); submit as many as there are GPUs to use.
# Keep the job name prefix plm_gen: the orchestration loops count workers by it (WORKER_JOB). Set --time to WORKER_SECS.
# Submit from the repository root (B = <work_dir>/benchmark):
#   benchmark:    sbatch --export=ALL,PLM_SAE_REPO="$PWD" paper/slurm/benchmark/gpu_worker.slurm
#   windowed:     sbatch --time=04:00:00 --export=ALL,PLM_SAE_REPO="$PWD",WORKER_SECS=14400,END_MARGIN=300,QLIST="$B/win:todo:2700" paper/slurm/benchmark/gpu_worker.slurm
#   progen2_t10:  sbatch --export=ALL,PLM_SAE_REPO="$PWD",END_MARGIN=300,QLIST="<work_dir>/progen2_t10:todo:2700" paper/slurm/benchmark/gpu_worker.slurm
# The ProGen2-OAS tasks use TF32 (Ampere or newer GPU); p-IgGen and IgLM tasks run on any GPU with 24 GB.
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
bash "$PLM_SAE_REPO/paper/scripts/benchmark/gpu_worker.sh"
