# Example SLURM jobs

Optional. Every stage runs without SLURM through the scripts in `scripts/` and `paper/scripts/`; these job scripts only
wrap them with resource requests. They contain no partition, cluster or log path. `examples/slurm/submit.sh` adds
those from the environment:

    export PLM_SAE_SLURM_GPU_PARTITION=gpu PLM_SAE_SLURM_CPU_PARTITION=cpu   # optional, as are the next three
    export PLM_SAE_SLURM_GPU_GRES=gpu:1 PLM_SAE_SLURM_CLUSTER=mycluster PLM_SAE_SLURM_LOG_DIR=/scratch/me/logs
    examples/slurm/submit.sh examples/slurm/<stage>/<job>.slurm [sbatch arguments]

The jobs source `scripts/env.sh` from `PLM_SAE_REPO` (exported by `submit.sh`), so set the data and work directories
in `configs/paths.yaml` or the environment before submitting.

Array jobs are idempotent: every task skips work whose output already exists, so a failed or preempted array can be
resubmitted unchanged. The resource requests are the ones used for the paper.
