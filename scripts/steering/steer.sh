#!/bin/bash
# Steer one SAE latent at a list of strengths and (optionally) number the libraries with ANARCI. One GPU.
# The steering vector is alpha times the latent's unit decoder vector, added at the final-layer-norm output (the read
# site of the hyperparameter-matched SAEs) at every position from START on. Sampling is the paper's
# (plm_sae.steering.generate): common random numbers across strengths, seeds 0 1 2, TOTAL valid sequences per library.
#
# Usage: scripts/steering/steer.sh MODEL SAE LATENT "ALPHAS" OUTDIR [START] [-- extra plm_sae.steering.generate flags]
#   MODEL   piggen | IgLM | progen2_oas
#   SAE     SAE name of the data record (e.g. piggen_batchtopk_r8) or a .safetensors file
#   LATENT  dictionary index of the latent
#   ALPHAS  steering strengths, e.g. "0 8 16 24 32 40 48" (include 0 for the unsteered reference)
#   START   first steered position (default 0: every position); the paper starts five positions before the latent's
#           onset in natural sequences (docs/protocol.md)
#   extra   e.g. -- --tf32 (the paper's ProGen2-OAS runs)
# Environment: SEEDS (default "0 1 2"), TOTAL (default 1000), NUMBER=0 to skip ANARCI, TARGET (label used in the
#   ANARCI file names, default none), NCPU (ANARCI processes, default 8).
# Output: OUTDIR/steered_l3_f<LATENT>_a<ALPHA>_seed<S>_seq<TOTAL>.txt (one sequence per line) and, with numbering,
#   the IMGT-numbered tables VH_alignment_..._formatted.csv next to them (heavy chain, germline J call).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -eo pipefail
[ $# -ge 5 ] || { sed -n '2,18p' "$0"; exit 1; }
MODEL=$1; SAE=$2; LAT=$3; ALPHAS=$4; OUT=$5; shift 5
START=0; if [ $# -gt 0 ] && [ "$1" != "--" ]; then START=$1; shift; fi
[ "${1:-}" = "--" ] && shift
SEEDS=${SEEDS:-0 1 2}; TOTAL=${TOTAL:-1000}; TARGET=${TARGET:-none}
mkdir -p "$OUT"; CFG="$OUT/configs.txt"; : > "$CFG"
for a in $ALPHAS; do echo "$LAT $a $START ." >> "$CFG"; done
log "steering $MODEL latent $LAT of $SAE at alphas [$ALPHAS], start $START -> $OUT"
"$PY" -m plm_sae.steering.generate --model "$MODEL" --sae "$SAE" --configs "$CFG" --site post --layer 3 \
  --seeds $SEEDS --total "$TOTAL" --min_yield 0.08 --outdir "$OUT" "$@"
[ "${NUMBER:-1}" = 0 ] && exit 0
if [ "$TOTAL" != 1000 ]; then log "ANARCI numbering expects 1,000-sequence libraries; skipped (TOTAL=$TOTAL)"; exit 0; fi
U="$OUT/units.txt"; : > "$U"
for a in $ALPHAS; do for s in $SEEDS; do printf "%s %s %s %.2f %s\n" "$(realpath "$OUT")" "$LAT" "$TARGET" "$a" "$s" >> "$U"; done; done
bash "$(dirname "${BASH_SOURCE[0]}")/../numbering/number_units.sh" "$U" "$OUT/anarci" "${NCPU:-8}"
log "libraries and ANARCI tables in $OUT"
