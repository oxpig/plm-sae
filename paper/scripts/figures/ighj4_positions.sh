#!/bin/bash
# Supplementary Figure 6: token and IMGT positions of the IGHJ4 latents 463, 4720 and 6276 of the expansion-32 TopK SAE
# over the 10,269 IGHJ4 validation heavy chains (paper.figures.fig_ighj4_positions). Each stage skips if its output
# exists. The encode stage needs a GPU (about 10 minutes; CPU works but takes hours): run it through
# paper/slurm/figures/ighj4_positions.slurm, or plot the deposited encode of the data record: STAGES=plot.
# Usage: paper/scripts/figures/ighj4_positions.sh [OUTDIR]           (default <work_dir>/figures)
# Inputs: <data_dir>/datasets/{master_Dec13_10.31.csv, vj_val_anarci.csv}, checkpoint piggen_topk_r32_layer3
# Writes: <work_dir>/figures/ighj4_positions/{ighj4_val.csv.gz, hits_pre_vh.npz},
#         OUTDIR/{absolute_positional_activations,high_res_IMGT_activations}_valonly_preln.png, OUTDIR/stats_valonly_preln.json
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -eo pipefail
OUT=${1:-$PLM_SAE_WORK_DIR/figures}
D="$PLM_SAE_WORK_DIR/figures/ighj4_positions"
for S in ${STAGES:-table encode plot}; do
  case $S in
    table) [ -f "$D/ighj4_val.csv.gz" ] || "$PY" -m paper.figures.fig_ighj4_positions table ;;
    encode) "$PY" -m paper.figures.fig_ighj4_positions encode ;;
    plot) "$PY" -m paper.figures.fig_ighj4_positions plot --out "$OUT" ;;
  esac
done
log "Supplementary Figure 6 done ($OUT)"
