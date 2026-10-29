#!/bin/bash
# FlashTAP scores of the benchmark libraries of the PSH-associated latents (Supplementary Section A.10): every alpha,
# seeds 0-2. ANARCI splits each p-IgGen output into VH|VL (main environment), FlashTAP scores the pairs
# ($PLM_SAE_PYTHON_FLASHTAP, GPU); then python -m paper.analyses.properties.dev_steering.analysis trend.
# Usage: paper/scripts/properties/dev_score.sh IDX      (0: TopK latent 980, 1: Batch TopK latent 1088)
# Reads   <work_dir>/benchmark/runs/piggen/<SAE>/dev/gated/dev/
# Writes  <work_dir>/benchmark/dev_scores/f<latent>/{pairs.csv, scores_f<latent>.csv}
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -euo pipefail
F="$PLM_SAE_WORK_DIR/benchmark"
case "${1:-0}" in 0) SAE=TopK; FEAT=980;; 1) SAE=BatchTopK; FEAT=1088;; *) echo "IDX 0 or 1" >&2; exit 2;; esac
WORK="$F/dev_scores/f$FEAT"; mkdir -p "$WORK"
for f in "$F/runs/piggen/$SAE/dev/gated/dev"/steered_l3_f${FEAT}_a*_seed[012]_seq1000.txt; do ln -sf "$f" "$WORK/"; done
"$PY" -m paper.analyses.properties.dev_steering.split_piggen_steered "$WORK" "$WORK/pairs.csv"
"$PLM_SAE_PYTHON_FLASHTAP" -m paper.analyses.properties.dev_steering.score_steered_sequences "$WORK/pairs.csv" "$FEAT" \
  "$WORK/scores_f$FEAT.csv"
