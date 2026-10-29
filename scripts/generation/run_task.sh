#!/bin/bash
# Run the generator for one steering task and file its libraries (the per-task body of every GPU worker of the paper:
# common random numbers, seeds 0 1 2, --min_yield 0.08, site post, layer 3 in the library file names).
#
# Usage: scripts/generation/run_task.sh TASKFILE [LOGDIR]
# The task file is sourced and sets
#   MODEL  piggen | IgLM | progen2_oas
#   CK     SAE name of the data record (e.g. piggen_batchtopk_r8), or a .safetensors file of steering directions
#   CFG    configs file of plm_sae.steering.generate, optionally followed by extra generator flags
#          (e.g. "--tf32" or "--site blockout")
#   OUT    library root; the generator's TAG subdirectories are moved below it
#   UNITS  lines "dir latent gene alpha seed": the libraries the task must produce
#   NLIB   number of libraries (informational)
# Exit status 0 if the generator succeeded and every library listed in UNITS exists. Writes LOGDIR/<task>.log and
# LOGDIR/<task>.generate_log.json (LOGDIR defaults to the task file's directory).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
TF=$(realpath "$1"); t=$(basename "$TF"); LOGDIR=${2:-$(dirname "$TF")}; mkdir -p "$LOGDIR"
source "$TF"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "${SLURM_JOB_ID:-local} $(hostname -s) $(date '+%F %T') gpu=$("$PY" -c 'import torch;print(torch.cuda.get_device_name(0))')" > "$LOGDIR/$t.info"
echo "$(date '+%T') START $t ($NLIB libraries) py=$PY"; s=$(date +%s)
TMP="$OUT/.generate_$t"; rm -rf "$TMP"
# $CFG is deliberately unquoted: it may carry extra generator flags after the file name
# (flags in $CFG come last, so a task's --site blockout overrides the default --site post)
"$PY" -m plm_sae.steering.generate --model "$MODEL" --sae "$CK" --site post --layer 3 --seeds 0 1 2 --min_yield 0.08 \
   --outdir "$TMP" --configs $CFG > "$LOGDIR/$t.log" 2>&1; rc=$?
if [ $rc -eq 0 ]; then
  (cd "$TMP" && find . -name 'steered_*.txt' | while read f; do mkdir -p "$OUT/$(dirname "$f")"; mv "$f" "$OUT/$f"; done)
  mv "$TMP/generate_log.json" "$LOGDIR/$t.generate_log.json" 2>/dev/null
fi
miss=0; while read d lat g a sd; do [ -f "$d/steered_l3_f${lat}_a${a}_seed${sd}_seq1000.txt" ] || miss=$((miss+1)); done < "$UNITS"
grep -h "WARNING" "$LOGDIR/$t.log" | cut -c1-300
if [ $rc -eq 0 ] && [ $miss -eq 0 ]; then echo "$(date '+%T') DONE $t $(( $(date +%s) - s ))s"; exit 0; fi
echo "$(date '+%T') FAILED $t rc=$rc missing=$miss"; tail -5 "$LOGDIR/$t.log"; exit 1
