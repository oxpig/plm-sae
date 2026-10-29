#!/bin/bash
# Generation, then ANARCI, of the tasks listed in <work_dir>/outsel_properties/tasks/<LIST>.lst: one array task per line
# (paper/slurm/properties/gen.slurm, at most 10 at once; paper/slurm/properties/anarci.slurm, at most 6 at once, each
# index after its generation index succeeded).
# Usage: paper/scripts/properties/submit_round.sh LIST      (base for the first round, then the ladder phases)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
S="$PLM_SAE_WORK_DIR/outsel_properties"; F="$S/tasks/$1.lst"; SUB="$PLM_SAE_REPO/examples/slurm/submit.sh"
[ -s "$F" ] || { echo "empty list $1"; exit 0; }
n=$(wc -w < "$F")
g=$(bash "$SUB" "$PLM_SAE_REPO/paper/slurm/properties/gen.slurm" --parsable --array=0-$((n - 1))%10 \
      --export=ALL,STATE="$S",LIST="$F" | cut -d';' -f1) || exit 1
a=$(bash "$SUB" "$PLM_SAE_REPO/paper/slurm/properties/anarci.slurm" --parsable --dependency=aftercorr:"$g" \
      --array=0-$((n - 1))%6 --export=ALL,STATE="$S",LIST="$F" | cut -d';' -f1) || exit 1
echo "$1: generation $g, ANARCI $a"
