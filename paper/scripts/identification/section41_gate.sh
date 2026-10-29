#!/bin/bash
# Feature identification gate of an expansion-32 TopK SAE of Section 4.1 (layer 3: Table 1, Supplementary Table 13;
# layers 0-3: Supplementary Table 12). paper.analyses.identification.section41_gate.
#   paper/scripts/identification/section41_gate.sh [LAYER]        (default 3)
# Reads   <work_dir>/activations/piggen_topk_r32_layer<L>/nonzero_mean/ (section41_encode.sh, all 32 shards)
# Writes  <work_dir>/identification/gates/piggen_topk_r32_layer<L>/ (full_model_evaluation_*, weights/,
#         feature_f1_scores_class_ighj<n>.csv, margin_sweep/)
# CPU only: 16 cores, 128 GB (layers 0-2 needed up to 340 GB), up to 6 h.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
NAME=piggen_topk_r32_layer${1:-3}
OUT="$PLM_SAE_WORK_DIR/identification/gates/$NAME"
if [ -f "$OUT/margin_sweep/summary.csv" ]; then log "done already: $OUT"; exit 0; fi
log "Section 4.1 gate $NAME -> $OUT"
"$PY" -m paper.analyses.identification.section41_gate "$PLM_SAE_ACTIVATIONS/$NAME/nonzero_mean" "$OUT"
cat "$OUT/margin_sweep/counts_by_margin.csv"
