#!/bin/bash
# Base-model activation cache for SAE training (plm_sae.activations.dump: LitData records of the final-layer-norm
# output, one per padded sequence). One GPU. Skips a cache whose index.json exists.
#
# Usage: scripts/extraction/dump_activations.sh piggen | IgLM | progen2_oas
# Output: $PLM_SAE_ACTIVATIONS/training/{piggen_layer3_100k, iglm_layer3, progen2oas_layer26} (the optimized_dir of
#         configs/training/datamodule/*.yaml). Disk: about 108 GB p-IgGen, 41 GB IgLM, 123 GB ProGen2-OAS.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
case "$1" in
  piggen) OUT=piggen_layer3_100k ;; IgLM) OUT=iglm_layer3 ;; progen2_oas) OUT=progen2oas_layer26 ;;
  *) echo "usage: dump_activations.sh piggen | IgLM | progen2_oas" >&2; exit 2 ;;
esac
OUT="$PLM_SAE_ACTIVATIONS/training/$OUT"
[ -f "$OUT/index.json" ] && { echo "exists: $OUT"; exit 0; }
# a LitData chunk cache per job: concurrent runs on one node collide in the default /tmp/chunks
export DATA_OPTIMIZER_CACHE_FOLDER="$PLM_SAE_WORK_DIR/tmp/litdata_chunks_${SLURM_JOB_ID:-$$}"
log "$1 activations -> $OUT"
"$PY" -m plm_sae.activations.dump "$1" "$OUT" || exit 1
rm -rf "$DATA_OPTIMIZER_CACHE_FOLDER"
log "done $1"
