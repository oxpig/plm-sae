#!/bin/bash
# Submit one of the job scripts in examples/slurm/ or paper/slurm/ with the site settings of the environment:
#   PLM_SAE_SLURM_GPU_PARTITION / PLM_SAE_SLURM_CPU_PARTITION   --partition (GPU when the script asks for --gres)
#   PLM_SAE_SLURM_GPU_GRES                                      --gres for GPU scripts (e.g. gpu:A100:1)
#   PLM_SAE_SLURM_CLUSTER                                       --clusters
#   PLM_SAE_SLURM_LOG_DIR                                       --output <dir>/%x_%A_%a.out (default <work_dir>/slurm_logs)
# Unset variables are left to the cluster's defaults. Extra arguments go to sbatch before the script, e.g.
#   examples/slurm/submit.sh paper/slurm/benchmark/gpu_step.slurm --export=ALL,STEP="paper.benchmark.positions piggen"
#   examples/slurm/submit.sh paper/slurm/benchmark/gpu_worker.slurm --array=0-3
# The job scripts source scripts/env.sh from PLM_SAE_REPO, which this exports.
# PLM_SAE_DRY_RUN=1 prints the sbatch command (and a placeholder job id for --parsable) instead of submitting.
set -e
[ $# -ge 1 ] || { sed -n '2,11p' "$0"; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/scripts/env.sh"
JOB=$1; shift
ARGS=()
if grep -q '^#SBATCH --gres' "$JOB"; then
  [ -n "${PLM_SAE_SLURM_GPU_PARTITION:-}" ] && ARGS+=(--partition="$PLM_SAE_SLURM_GPU_PARTITION")
  [ -n "${PLM_SAE_SLURM_GPU_GRES:-}" ] && ARGS+=(--gres="$PLM_SAE_SLURM_GPU_GRES")
else
  [ -n "${PLM_SAE_SLURM_CPU_PARTITION:-}" ] && ARGS+=(--partition="$PLM_SAE_SLURM_CPU_PARTITION")
fi
[ -n "${PLM_SAE_SLURM_CLUSTER:-}" ] && ARGS+=(--clusters="$PLM_SAE_SLURM_CLUSTER")
LOGS=${PLM_SAE_SLURM_LOG_DIR:-$PLM_SAE_WORK_DIR/slurm_logs}
export PLM_SAE_REPO="$REPO"
if [ "${PLM_SAE_DRY_RUN:-0}" = 1 ]; then
  echo "sbatch ${ARGS[*]} --output=$LOGS/%x_%A_%a.out $* $JOB" >&2
  echo "DRY_$(basename "$JOB" .slurm)"
  exit 0
fi
mkdir -p "$LOGS"
sbatch "${ARGS[@]}" --output="$LOGS/%x_%A_%a.out" "$@" "$JOB"
