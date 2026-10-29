#!/bin/bash
# IMGT localisation (Supplementary Table 14, the data of Supplementary Figures 8 and 9, the "five of six" of
# Section 4.3). CPU, 40 processes.
# Needs the IMGT table datasets/master_Dec13_10.31.csv, the per-token validation latents of the p-IgGen SAEs
# (<work_dir>/activations/piggen_*_r8/val/) and the IGHJ4 token latents of the IgLM and ProGen2-OAS TopK SAEs
# (<work_dir>/activations/{iglm,progen2oas}_topk_r8/val_ighj4_tok/; paper/scripts/figures/encode_ighj4_tokens.sh).
# Writes <work_dir>/identification/imgt/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
"$PY" -m paper.analyses.identification.localisation
"$PY" -m paper.analyses.identification.imgt_position_counts
