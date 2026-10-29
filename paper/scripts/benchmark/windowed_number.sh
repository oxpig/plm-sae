#!/bin/bash
# Numbering loop of the windowed-steering tasks (state <work_dir>/benchmark/win): every finished task in win/tasks/done not
# yet queued is numbered in win/anarci/<task> (deduplicated ANARCI, 4 array jobs x 8 CPUs, 16 GB / 32 GB, as in the paper's
# windowed run).
# Ends when no windowed task is queued or running and every finished task is queued for numbering, or when no GPU worker
# (SLURM job-name prefix WORKER_JOB, default plm_gen) is left.
# Run in the background:  bash paper/scripts/benchmark/windowed_number.sh > <work_dir>/benchmark/win/logs/number.log 2>&1
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1; source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
WF=$PLM_SAE_WORK_DIR/benchmark/win; cd "$WF" || exit 1; mkdir -p anarci logs/anarci
export ANARCI_NUM_MEM=${ANARCI_NUM_MEM:-16G} ANARCI_ASM_MEM=${ANARCI_ASM_MEM:-32G}
WJ=${WORKER_JOB:-plm_gen}
while true; do
  for t in $(tasks_in tasks/done); do
    [ -e "anarci/$t.queued" ] && continue
    number_task "$WF" "$t" 4 8 && echo "$(ts) queued $t"
  done
  left=$(tasks_in tasks/todo tasks/claimed | wc -l)
  unq=$(for t in $(tasks_in tasks/done); do [ -e "anarci/$t.queued" ] || echo "$t"; done | wc -l)
  [ "$left" = 0 ] && [ "$unq" = 0 ] && { echo "$(ts) all tasks queued for numbering"; break; }
  have_slurm && [ "$(njobs "$WJ")" = 0 ] && [ "$unq" = 0 ] && { echo "$(ts) no workers left; unfinished: $left"; break; }
  sleep 120
done
