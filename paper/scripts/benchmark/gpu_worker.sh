#!/bin/bash
# GPU worker of the steering pipelines (benchmark, windowed, progen2_t10). Claims the lowest-sorted task file from an
# ordered list of STATE:QUEUE pairs (atomic mv into STATE/tasks/claimed), runs it with scripts/generation/run_task.sh
# (fp32, common random numbers, seeds 0 1 2, --min_yield 0.08), then moves it with its .log, .info and
# .generate_log.json to STATE/tasks/done, or to STATE/tasks/failed. No automatic retries. A task is started only while
# at least MINLEFT seconds of the worker's time remain (tasks are sized to finish within that). Exits when nothing is
# claimable and every listed STATE has ALL_TASKS_ADDED, when too little time is left, or (after the current task) when
# a file STOP exists in a listed STATE.
#
# Usage: paper/scripts/benchmark/gpu_worker.sh [STATE:QUEUE[:MINLEFT] ...]       (or env QLIST holding the same list)
#   STATE = pipeline state directory, QUEUE = directory below STATE/tasks (default <work_dir>/benchmark:todo).
# Environment: WORKER_SECS (worker time in seconds, default 43200), END_MARGIN (600), MINLEFT (default for entries
#   without one, 7200). Workers of each pipeline (B = <work_dir>/benchmark):
#   benchmark    (no list)                                WORKER_SECS=43200 END_MARGIN=600
#   windowed     "$B/win:todo:2700"                       WORKER_SECS=14400 END_MARGIN=300
#   progen2_t10  "<work_dir>/progen2_t10:todo:2700"       WORKER_SECS=43200 END_MARGIN=300 (TF32: Ampere or newer GPU)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
RUN="$PLM_SAE_REPO/scripts/generation/run_task.sh"
W=${SLURM_JOB_ID:-local}; END=$(( $(date +%s) + ${WORKER_SECS:-43200} - ${END_MARGIN:-600} ))
GNAME=$("$PY" -c 'import torch;print(torch.cuda.get_device_name(0))') || exit 1
LIST=("$@"); [ ${#LIST[@]} -eq 0 ] && read -ra LIST <<< "${QLIST:-}"
[ ${#LIST[@]} -eq 0 ] && LIST=("$PLM_SAE_WORK_DIR/benchmark:todo")
log "worker $W on $(hostname -s) gpu=$GNAME queues: ${LIST[*]}"
while true; do
  for e in "${LIST[@]}"; do [ -f "${e%%:*}/STOP" ] && { log "STOP"; exit 0; }; done
  t=""; S=""; minml=""
  for e in "${LIST[@]}"; do
    IFS=: read -r s q ml <<< "$e"; ml=${ml:-${MINLEFT:-7200}}
    if [ -z "$minml" ] || [ "$ml" -lt "$minml" ]; then minml=$ml; fi
    [ $(( END - $(date +%s) )) -lt "$ml" ] && continue
    mkdir -p "$s/tasks/claimed" "$s/tasks/done" "$s/tasks/failed"
    for c in $(ls "$s/tasks/$q" 2>/dev/null | sort); do
      if mv "$s/tasks/$q/$c" "$s/tasks/claimed/$c" 2>/dev/null; then t=$c; S=$s; break 2; fi
    done
  done
  if [ -z "$t" ]; then
    all=1; for e in "${LIST[@]}"; do [ -f "${e%%:*}/ALL_TASKS_ADDED" ] || all=0; done
    if [ $all = 1 ] || [ $(( END - $(date +%s) )) -lt "$minml" ]; then log "no more tasks"; break; fi
    sleep 60; continue
  fi
  T=$S/tasks
  if bash "$RUN" "$T/claimed/$t" "$T/claimed"; then
    mv "$T/claimed/$t" "$T/done/"; mv "$T/claimed/$t.log" "$T/claimed/$t.info" "$T/claimed/$t.generate_log.json" "$T/done/" 2>/dev/null
  else
    mv "$T/claimed/$t" "$T/failed/"; mv "$T/claimed/$t.log" "$T/claimed/$t.info" "$T/failed/" 2>/dev/null
  fi
done
