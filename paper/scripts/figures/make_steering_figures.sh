#!/bin/bash
# J-gene steering figures from the deposited library tables (paper.figures.fig_steering; CPU, under a minute each).
#   fig2_f1.png         Figure 2: p-IgGen, (a) the passing latent ranked first by F1 with the largest rise (Batch TopK
#                       latent 733, gated) and (b) the latent ranked first by output selection for the same gene (TopK
#                       latent 5141), IGHJ6
#   fig_xmodel.png      Figure 3: the passing latent ranked first by F1 with the largest rise in IgLM (Ordered latent 1214,
#                       IGHJ5, without the position gate) and ProGen2-OAS (Batch TopK latent 7567, IGHJ3, temperature 1.0)
#   fig_case_steer.png  Supplementary Figure 7: expansion-32 TopK latents 463, 4720 and 6276 (IGHJ4)
# Reads benchmark and progen2_t10 results (<work_dir> where present, else <results_dir>); writes OUTDIR (default
# <work_dir>/figures).
# Usage: paper/scripts/figures/make_steering_figures.sh [OUTDIR]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
O="${1:-$PLM_SAE_WORK_DIR/figures}"; mkdir -p "$O"
"$PY" -m paper.figures.fig_steering "$O/fig2_f1.png" 3.3 1.08 2 \
  '(a) Top $F_1$ 733|benchmark|piggen|BatchTopK/f1/gated|733|IGHJ6' \
  '(b) Output-selected 5141|benchmark|piggen|TopK/output/gated|5141|IGHJ6'
"$PY" -m paper.figures.fig_steering "$O/fig_xmodel.png" 3.3 1.08 2 \
  '(a) IgLM, top $F_1$ 1214|benchmark|IgLM|Ordered/f1/ungated|1214|IGHJ5' \
  '(b) ProGen2-OAS, top $F_1$ 7567|progen2_t10|progen2_oas|BatchTopK/f1|7567|IGHJ3'
"$PY" -m paper.figures.fig_steering "$O/fig_case_steer.png" 6.9 1.75 3 \
  "(a) Latent 463|benchmark|piggen|TopK/section41/gated|463|IGHJ4" \
  "(b) Latent 4720|benchmark|piggen|TopK/section41/gated|4720|IGHJ4" \
  "(c) Latent 6276|benchmark|piggen|TopK/section41/gated|6276|IGHJ4"
