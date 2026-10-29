#!/bin/bash
# Ladder loop of the ProGen2-OAS benchmark (temperature 1.0): every 5 min, progen2_t10/ladder.py queues extension / refinement tasks for
# latents whose tasks are all done and numbered. When all 162 latents are final: ALL_TASKS_ADDED (workers exit), final
# scoring (progen2_t10/score.py -> score/), T10_DONE.
# Run in the background:  bash paper/scripts/benchmark/progen2_t10_ladder.sh > <work_dir>/progen2_t10/logs/ladder.log 2>&1
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
B=$PLM_SAE_WORK_DIR/progen2_t10; cd "$B" || exit 1; mkdir -p score
while true; do
  out=$(NPROC=8 "$PY" -m paper.benchmark.progen2_t10.ladder 2>&1 | grep -v Warning); echo "$(date '+%F %T') $(echo "$out" | tail -n 1)"; echo "$out" | grep '^queued'
  if echo "$out" | tail -n 1 | grep -q "QUEUED 0 PENDING 0 FINAL 162"; then
    touch ALL_TASKS_ADDED; NPROC=16 "$PY" -m paper.benchmark.progen2_t10.score > score/final.log 2>&1; echo "$(date '+%F %T') final scoring rc=$?"; touch T10_DONE; break
  fi
  sleep 300
done
