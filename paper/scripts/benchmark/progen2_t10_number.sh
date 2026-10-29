#!/bin/bash
# Numbering loop of the ProGen2-OAS benchmark (temperature 1.0) (state <work_dir>/progen2_t10): deduplicated ANARCI (1 array job x 16
# CPUs) of every finished task in tasks/done, work directory anarci/<task>. Stops when no GPU worker (SLURM job-name prefix
# WORKER_JOB, default plm_gen) is left and every finished task is numbered; without SLURM, when ALL_TASKS_ADDED exists
# (written by progen2_t10_ladder.sh) and every finished task is numbered.
# Run in the background:  bash paper/scripts/benchmark/progen2_t10_number.sh > <work_dir>/progen2_t10/logs/number.log 2>&1
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1; source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
B=$PLM_SAE_WORK_DIR/progen2_t10; cd "$B" || exit 1; mkdir -p anarci logs/anarci
WJ=${WORKER_JOB:-plm_gen}
while true; do
  for t in $(tasks_in tasks/done); do
    [ -e "anarci/$t.queued" ] && continue
    number_task "$B" "$t" 1 16; echo "$(ts) numbering $t"
  done
  left=$(tasks_in tasks/todo tasks/claimed | wc -l)
  un=0; for t in $(tasks_in tasks/done); do numbered "anarci/$t" || un=$((un+1)); done
  if have_slurm; then [ "$(njobs "$WJ")" = 0 ] && [ $un = 0 ] && { echo "$(ts) no workers, all numbered (left in queue: $left)"; break; }
  else [ -f ALL_TASKS_ADDED ] && [ $un = 0 ] && { echo "$(ts) all numbered"; break; }; fi
  sleep 120
done
