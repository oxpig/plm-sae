#!/bin/bash
# Pooled latents of an expansion-32 TopK SAE of Section 4.1 (piggen_topk_r32_layer<L>) for one shard of a labelled
# split: the SAE reads the output of its p-IgGen block; per latent, the mean of its non-zero activations over the
# residue positions (plm_sae.activations.encode --pool nonzero_mean). Input of the Section 4.1 gate (Table 1,
# Supplementary Tables 12 and 13). One GPU, under 1.5 hours per shard.
#
# Usage: paper/scripts/identification/section41_encode.sh TASK [LAYER]
#   TASK   0-31: 0-7 validation shards (8 in all), 8-31 training shards (24 in all)
#   LAYER  0-3 (default 3)
# Output: $PLM_SAE_ACTIVATIONS/piggen_topk_r32_layer<LAYER>/nonzero_mean/{train,val}_shardNNN.npz + _rows.npy
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
T=$1; LAYER=${2:-3}
if [ "$T" -lt 8 ]; then SPLIT=val; SH=$T; NS=8; else SPLIT=train; SH=$((T - 8)); NS=24; fi
NAME=piggen_topk_r32_layer$LAYER; OUT="$PLM_SAE_ACTIVATIONS/$NAME/nonzero_mean"
F="$OUT/${SPLIT}_shard$(printf '%03d' "$SH")_rows.npy"
[ -f "$F" ] && { echo "exists: $F"; exit 0; }
log "encode $NAME $SPLIT shard $SH/$NS -> $OUT"
"$PY" -m plm_sae.activations.encode "$NAME" "$PLM_SAE_DATA_DIR/datasets/vj_${SPLIT}_anarci.csv" "$OUT" \
    --pool nonzero_mean --split "$SPLIT" --shard "$SH" --num_shards "$NS" || exit 1
