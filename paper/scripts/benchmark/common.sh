# Shared functions of the benchmark orchestration scripts. Source after scripts/env.sh:
#   source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" && source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# Numbering: a SLURM array per task (paper/scripts/benchmark/anarci_array.sh), or with NUMBER_LOCAL=1 in the foreground on this
# machine (scripts/numbering/number_units.sh). Job-name prefixes: anarci_ (numbering, set by anarci_array.sh) and WORKER_JOB
# (GPU workers; default plm_gen, the job name used in paper/slurm/benchmark/gpu_worker.slurm).
ts() { date '+%F %T'; }
have_slurm() { command -v squeue >/dev/null 2>&1; }
# number of this user's queued or running SLURM jobs whose name starts with $1 (0 without SLURM)
njobs() { have_slurm || { echo 0; return; }
  squeue ${PLM_SAE_SLURM_CLUSTER:+-M "$PLM_SAE_SLURM_CLUSTER"} -u "${USER:-$(id -un)}" -h -o %j | grep -c "^$1"; }
units_of() { grep '^UNITS=' "$1" | cut -d= -f2 | tr -d '"'; }
# number_task STATE TASK NT C: number the libraries of STATE/tasks/done/TASK in STATE/anarci/TASK (marker HI_DONE) and mark
# it STATE/anarci/TASK.queued; log STATE/logs/anarci/TASK.log
number_task() { local S=$1 t=$2 nt=$3 c=$4 U; U=$(units_of "$S/tasks/done/$t"); [ -s "$U" ] || return
  mkdir -p "$S/anarci" "$S/logs/anarci"
  if [ "${NUMBER_LOCAL:-0}" = 1 ]; then
    bash "$PLM_SAE_REPO/scripts/numbering/number_units.sh" "$U" "$S/anarci/$t" "$c" > "$S/logs/anarci/$t.log" 2>&1
  else
    bash "$PLM_SAE_REPO/paper/scripts/benchmark/anarci_array.sh" "$U" "$S/anarci/$t" "${t:0:14}_$(echo "$t" | md5sum | cut -c1-4)" \
      "$nt" "$c" > "$S/logs/anarci/$t.log" 2>&1
  fi
  touch "$S/anarci/$t.queued"; }
numbered() { [ -f "$1/assemble.log" ] && grep -q HI_DONE "$1/assemble.log"; }
# task files (no dot in the name) in the given queue directories
tasks_in() { ls "$@" 2>/dev/null | grep -v '\.' | grep -v ':$' | grep -v '^$'; }
