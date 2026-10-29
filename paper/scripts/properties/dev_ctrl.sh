#!/bin/bash
# FlashTAP PSH of the baseline latents compared with the developability latents (dev_ctrl_list.csv, from
# python -m paper.analyses.properties.dev_steering.analysis controls): each at its valid_amax, seeds 0-2. Baseline i goes
# to array index i mod 6; scored baselines are skipped.
# Usage: paper/scripts/properties/dev_ctrl.sh IDX      (IDX 0..5)
# Writes <work_dir>/benchmark/dev_scores/ctrl_<latent>/{pairs.csv, scores_f<latent>.csv}
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -euo pipefail
F="$PLM_SAE_WORK_DIR/benchmark"; IDX=${1:-0}; i=0
tail -n +2 "$F/dev_ctrl_list.csv" | while IFS=, read arch target latent start alpha dir; do
  if [ $(( i % 6 )) -eq "$IDX" ]; then
    WORK="$F/dev_scores/ctrl_$latent"; mkdir -p "$WORK"
    if [ ! -s "$WORK/scores_f$latent.csv" ]; then
      a=$(printf "%.2f" "$alpha")
      for s in 0 1 2; do ln -sf "$dir/steered_l3_f${latent}_a${a}_seed${s}_seq1000.txt" "$WORK/"; done
      "$PY" -m paper.analyses.properties.dev_steering.split_piggen_steered "$WORK" "$WORK/pairs.csv"
      "$PLM_SAE_PYTHON_FLASHTAP" -m paper.analyses.properties.dev_steering.score_steered_sequences "$WORK/pairs.csv" \
        "$latent" "$WORK/scores_f$latent.csv"
    fi
  fi
  i=$(( i + 1 ))
done
