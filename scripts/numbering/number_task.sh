#!/bin/bash
# ANARCI numbering of one finished generation task of a pipeline state directory (STATE/tasks/done/<task>), on one node:
# scripts/numbering/number_units.sh on the task's units file, work directory STATE/anarci/<task> (marker HI_DONE).
# Usage: scripts/numbering/number_task.sh STATE TASK [NCPU]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S=$(realpath "$1"); t=$2; W=$S/anarci/$t
[ -f "$W/HI_DONE" ] && { echo "$t already numbered"; exit 0; }
[ -f "$S/tasks/done/$t" ] || { echo "task $t not done"; exit 1; }
U=$(grep '^UNITS=' "$S/tasks/done/$t" | cut -d= -f2 | tr -d '"')
bash "$HERE/number_units.sh" "$U" "$W" "${3:-${SLURM_CPUS_PER_TASK:-32}}"
