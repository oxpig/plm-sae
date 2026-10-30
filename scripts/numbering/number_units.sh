#!/bin/bash
# Deduplicated ANARCI numbering (IMGT scheme, germline assignment) of the libraries listed in a units file, on one
# node: collect the unique sequences, number them on NCPU processes (hmmscan single-threaded), then write each
# library's VH_alignment CSV and _formatted CSV next to it (plm_sae.numbering.number). Libraries whose formatted
# CSV already exists are skipped. Marker: WORKDIR/HI_DONE.
#
# Usage: scripts/numbering/number_units.sh UNITS WORKDIR [NCPU]     (UNITS lines: "dir latent gene alpha seed")
# For many libraries, paper/scripts/benchmark/anarci_array.sh spreads the number step over several nodes.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -o pipefail   # a failed step must stop here (each step's output goes through tee), before HI_DONE is written
U=$(realpath "$1"); W=$2; C=${3:-${SLURM_CPUS_PER_TASK:-32}}
[ -f "$W/HI_DONE" ] && { echo "$W already numbered"; exit 0; }
mkdir -p "$W"; rm -f "$W"/res_*.pkl
export PATH="$(dirname "$PY"):$PATH" HMMER_NCPU=1   # hmmscan from the environment's bin directory
"$PY" -m plm_sae.numbering.number collect "$U" "$W" | tee "$W/collect.log" || exit 1
"$PY" -m plm_sae.numbering.number number "$W" 0 1 "$C" | tee "$W/number_0.log" || exit 1
"$PY" -m plm_sae.numbering.number assemble "$U" "$W" "$C" | tee "$W/assemble.log" || exit 1
echo HI_DONE >> "$W/assemble.log"; touch "$W/HI_DONE"; echo "$(date '+%T') numbered $U"
