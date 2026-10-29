#!/bin/bash
# FVU, L0 and dead fraction on the 30,754 validation sequences (Supplementary Tables 4 and 5; per-latent firing counts
# reused by the redundancy stage).
#   scripts/identification/run_reconstruction.sh {piggen|IgLM|progen2_oas|piggen_exp32}
# One GPU, under 2 h each. Writes <work_dir>/identification/reconstruction/{recon_val,fire_counts}_<key>.*
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
KEY=${1:?usage: run_reconstruction.sh piggen|IgLM|progen2_oas|piggen_exp32}
OUT="$PLM_SAE_WORK_DIR/identification/reconstruction/recon_val_$KEY.csv"
if [ -f "$OUT" ]; then log "done already: $OUT"; exit 0; fi
"$PY" -m plm_sae.training.reconstruction "$KEY"
