#!/bin/bash
# Benchmark loop with the adaptive ladder (Supplementary Section A.2), after `python -m paper.benchmark.make_tasks` queued
# the sets: ANARCI numbering of every finished task (2 array jobs x 32 CPUs per task) -> paper.benchmark.extend
# (extension tasks x<block>_<parent> for latents whose pooled coverage at the last alpha is still >= 80%) -> once nothing is queued,
# running, numbering or awaiting evaluation: ALL_TASKS_ADDED (workers exit when their queues are empty) and
# paper.benchmark.analyze_final (final_latents.csv). Tasks queued while it runs are numbered and extended by the same loop.
# Stops early, reporting what is left, if no GPU worker (SLURM job-name prefix WORKER_JOB, default plm_gen) and no
# numbering job is left.
# Run in the background on a CPU node or as a long CPU job:
#   bash paper/scripts/benchmark/chain.sh > <work_dir>/benchmark/logs/chain.log 2>&1
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1; source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
F=$PLM_SAE_WORK_DIR/benchmark; cd "$F" || exit 1; mkdir -p anarci logs/anarci
WJ=${WORKER_JOB:-plm_gen}
to_number() { tasks_in tasks/done | while read t; do [ -e "anarci/$t.queued" ] || echo "$t"; done; }
echo "$(ts) chain start"
while true; do
  for t in $(to_number); do number_task "$F" "$t" 2 32; done
  out=$("$PY" -m paper.benchmark.extend 2>>logs/extend.err); echo "$out" | grep -v "^PENDING" | sed "s/^/$(ts) /" >> logs/extend.log
  echo "$out" | grep "^queued" | sed "s/^/$(ts) /"
  pend=$(echo "$out" | awk '/^PENDING/{print $2}'); pend=${pend:-1}
  gen=$(tasks_in tasks/todo tasks/claimed | wc -l); an=$(njobs anarci_); workers=$(njobs "$WJ")
  if [ "$gen" = 0 ] && [ "$an" = 0 ] && [ "$pend" = 0 ] && [ "$(to_number | wc -l)" = 0 ]; then break; fi
  if have_slurm && [ "$workers" = 0 ] && [ "$an" = 0 ]; then
    echo "$(ts) no workers left; unfinished: $(tasks_in tasks/todo tasks/claimed | tr '\n' ' ')"; break; fi
  sleep 120
done
touch ALL_TASKS_ADDED; echo "$(ts) generation, numbering and extension complete"
echo "$(ts) ANARCI problems: $(cat anarci/*/assemble.log 2>/dev/null | grep -h 'assemble:' | awk '{s+=$5} END{print s}') units without numbered chains (collapsed libraries)"
echo "$(ts) failed tasks: $(tasks_in tasks/failed | tr '\n' ' ')"
"$PY" -m paper.benchmark.analyze_final > logs/analyze_final.log 2>&1; echo "$(ts) analyze_final rc=$?"
echo "$(ts) CHAIN_DONE"
