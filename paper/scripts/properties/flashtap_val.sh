#!/bin/bash
# FlashTAP scores (PSH, PPC, PNC, SFvCSP) of the 30,754 validation pairs (VH|VL of vj_val_anarci.csv), the labels of the
# developability gate. One GPU, $PLM_SAE_PYTHON_FLASHTAP. Skips if the scores exist.
# Output: <work_dir>/developability/flashtap/flashtap_val_scores.csv
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -eo pipefail
OUT="$PLM_SAE_WORK_DIR/developability/flashtap"; mkdir -p "$OUT"
[ -s "$OUT/flashtap_val_scores.csv" ] && { echo "done already: $OUT/flashtap_val_scores.csv"; exit 0; }
"$PLM_SAE_PYTHON_FLASHTAP" -m paper.analyses.properties.developability.flashtap_val \
  "$PLM_SAE_DATA_DIR/datasets/vj_val_anarci.csv" "$OUT/flashtap_val_scores.csv"
