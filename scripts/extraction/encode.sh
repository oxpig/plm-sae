#!/bin/bash
# Latents of a hyperparameter-matched SAE for every labelled sequence of one split: the input of feature
# identification (plm_sae.activations.encode; heavy chain, final-layer-norm output / norm factor). One GPU; resumable.
#   p-IgGen SAEs        per-token files [350, 6144] (also read by the IMGT localisation)
#   IgLM, ProGen2-OAS   mean over all 200 padded positions
# Batch TopK and Ordered SAEs are encoded one sequence per call, TopK SAEs of IgLM and ProGen2-OAS 64 per call.
#
# Usage: scripts/extraction/encode.sh NAME SPLIT [vhvl]
#   NAME   piggen_{topk,batchtopk,ordered}_r8 | iglm_{...}_r8 | progen2oas_{...}_r8
#   SPLIT  train (103,055 sequences) | val (30,754)
#   vhvl   p-IgGen light-chain-type gate (Supplementary Section A.7): paired VH+VL of vj_<SPLIT>_anarci_lct.csv
#          (datasets, from paper.analyses.identification.add_light_chain_type), a seed-42 sample of 6,000 training or 4,000 validation sequences
# Output: $PLM_SAE_ACTIVATIONS/NAME/SPLIT[_vhvl]/<orig_row>.npy. Disk: p-IgGen about 8.6 MB per sequence (265 GB for
# val, 890 GB for train).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
NAME=$1; SPLIT=$2; VARIANT=${3:-}
case "$SPLIT" in train|val) ;; *) echo "SPLIT must be train or val" >&2; exit 2 ;; esac
CSV="$PLM_SAE_DATA_DIR/datasets/vj_${SPLIT}_anarci.csv"; OUT="$PLM_SAE_ACTIVATIONS/$NAME/$SPLIT"; ARGS=()
case "$NAME" in
  piggen_*) ARGS=(--pool tokens) ;;
  iglm_topk_r8|progen2oas_topk_r8) ARGS=(--pool mean --batch_size 64) ;;
  iglm_*|progen2oas_*) ARGS=(--pool mean) ;;
  *) echo "unknown SAE $NAME" >&2; exit 2 ;;
esac
if [ "$VARIANT" = vhvl ]; then
  N=6000; [ "$SPLIT" = val ] && N=4000
  CSV="$PLM_SAE_DATA_DIR/datasets/vj_${SPLIT}_anarci_lct.csv"; OUT="${OUT}_vhvl"
  ARGS+=(--seq_col "VH+VL" --sample "$N" --seed 42)
fi
[ -f "$CSV" ] || { echo "missing $CSV" >&2; exit 1; }
log "encode $NAME $SPLIT ${VARIANT:-VHorVHH} -> $OUT"
"$PY" -m plm_sae.activations.encode "$NAME" "$CSV" "$OUT" "${ARGS[@]}" || exit 1
log "done $(find "$OUT" -name '*.npy' | wc -l) files"
