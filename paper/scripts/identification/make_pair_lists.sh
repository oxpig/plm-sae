#!/bin/bash
# Latent ranking lists of the steering benchmark, from finished gates (plm_sae.features.rank; CPU, seconds):
#   pairs_<sae>_top5.txt  the five highest-F1 latents per J gene, nine matched SAEs (ranks 1-3 are steered and reported;
#                         ranks 4-5 are excluded from the baseline draw)
#   pairs_<sae>_lct.txt   the three highest-F1 latents per light-chain type, p-IgGen SAEs
#   rank_latents_463_4720_6276.csv  F1 ranks of three IGHJ4 latents of the Section 4.1 SAE (Supplementary Section A.5)
# Writes <work_dir>/identification/pairs/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
OUT="$PLM_SAE_WORK_DIR/identification/pairs"; mkdir -p "$OUT"
G="$PLM_SAE_WORK_DIR/identification/gates"
for N in piggen_topk_r8 piggen_batchtopk_r8 piggen_ordered_r8 iglm_topk_r8 iglm_batchtopk_r8 iglm_ordered_r8 \
         progen2oas_topk_r8 progen2oas_batchtopk_r8 progen2oas_ordered_r8; do
  "$PY" -m plm_sae.features.rank "$G/$N" --top_n 5 > "$OUT/pairs_${N}_top5.txt"
done
for N in piggen_topk_r8 piggen_batchtopk_r8 piggen_ordered_r8; do
  "$PY" -m plm_sae.features.rank "$G/${N}_lct" --top_n 3 --classes kappa lambda > "$OUT/pairs_${N}_lct.txt"
done
"$PY" -m plm_sae.features.rank --rank_of "$G/piggen_topk_r32_layer3/feature_f1_scores_class_ighj4.csv" \
  --latents 463 4720 6276 --out "$OUT/rank_latents_463_4720_6276.csv"
log "wrote $OUT"
