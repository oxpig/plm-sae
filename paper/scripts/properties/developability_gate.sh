#!/bin/bash
# Developability gate of one hyperparameter-matched SAE (Supplementary Section A.10, Supplementary Table 19): Pearson
# correlation of every latent's pooled validation activation with the four FlashTAP metrics (after flashtap_val.sh and
# scripts/extraction/encode.sh SAE val). CPU: 8 cores, 48 GB, up to 10 h. Skips if the four tables exist.
# Usage: paper/scripts/properties/developability_gate.sh SAE      (piggen_topk_r8, ..., progen2oas_ordered_r8)
# Output: <work_dir>/developability/gate/SAE/developability_<metric>_per_latent.csv
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -eo pipefail
SAE=${1:?usage: developability_gate.sh SAE}
OUT="$PLM_SAE_WORK_DIR/developability/gate/$SAE"
[ "$(ls "$OUT"/developability_*_per_latent.csv 2>/dev/null | wc -l)" -ge 4 ] && { echo "$SAE done already"; exit 0; }
"$PY" -m paper.analyses.properties.developability.gate "$SAE"
