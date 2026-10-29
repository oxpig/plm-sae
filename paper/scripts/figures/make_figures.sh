#!/bin/bash
# Every figure of the paper from the result tables (CPU). Writes OUTDIR (default <work_dir>/figures) with the file
# names of the manuscript.
#   Figure 1, Supplementary Figure 5    paper.figures.fig_latent_profiles (one p-IgGen forward pass; needs checkpoint
#                                       piggen_topk_r32_layer3)
#   Figures 2 and 3, Supp. Figure 7     paper/scripts/figures/make_steering_figures.sh
#   Supplementary Figure 4              paper.analyses.quality.by_set pool, then paper.figures.fig_quality_violins
#   Supplementary Figure 6              paper/scripts/figures/ighj4_positions.sh, plot stage (a fresh encode:
#                                       STAGES="table encode plot", paper/slurm/figures/ighj4_positions.slurm)
#   Supplementary Figures 8 and 9       paper.figures.fig_imgt
# Usage: paper/scripts/figures/make_figures.sh [OUTDIR]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
O="${1:-$PLM_SAE_WORK_DIR/figures}"; mkdir -p "$O"
bash "$(dirname "${BASH_SOURCE[0]}")/make_steering_figures.sh" "$O"
"$PY" -m paper.figures.fig_imgt "$O"
"$PY" -m paper.analyses.quality.by_set pool
"$PY" -m paper.figures.fig_quality_violins "$O/supp_quality_violins.png"
"$PY" -m paper.figures.fig_latent_profiles "$O"
STAGES=plot bash "$(dirname "${BASH_SOURCE[0]}")/ighj4_positions.sh" "$O"
log "figures in $O"
