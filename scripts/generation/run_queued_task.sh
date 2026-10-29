#!/bin/bash
# One generation task of a pipeline state directory, without a worker queue: the task file is STATE/tasks/<task>; on
# success it moves to STATE/tasks/done (with its .log, .info and .generate_log.json), otherwise a copy goes to
# STATE/tasks/failed. Idempotent: exits 0 at once if the task is already done and every library exists.
# Usage: scripts/generation/run_queued_task.sh STATE TASK      (generation itself: scripts/generation/run_task.sh)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S=$(realpath "$1"); T=$S/tasks; t=$2
if [ -f "$T/done/$t" ]; then src=$T/done/$t; elif [ -f "$T/$t" ]; then src=$T/$t; else echo "no task file $t"; exit 1; fi
UNITS=$(grep '^UNITS=' "$src" | cut -d= -f2 | tr -d '"')
nmiss() { local m=0; while read d lat g a sd; do [ -f "$d/steered_l3_f${lat}_a${a}_seed${sd}_seq1000.txt" ] || m=$((m+1)); done < "$UNITS"; echo $m; }
[ "$src" = "$T/done/$t" ] && [ "$(nmiss)" -eq 0 ] && { echo "$t already done"; exit 0; }
mkdir -p "$T/done" "$T/failed"
if bash "$HERE/run_task.sh" "$src" "$T"; then
  [ "$src" = "$T/done/$t" ] || mv "$src" "$T/done/"
  mv "$T/$t.log" "$T/$t.info" "$T/$t.generate_log.json" "$T/done/" 2>/dev/null; exit 0
else
  cp "$src" "$T/failed/"; exit 1
fi
