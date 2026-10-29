#!/bin/bash
# Learning-rate screens of the hyperparameter-matched SAEs (Methods): 10,000 steps at each learning rate of the grid
# 1e-7 1e-6 1e-5 1e-4 1e-3, otherwise the settings of the full run (sae_runs.sh), then FVU, L0 and dead fraction of
# each screen SAE on the first 128 sequences of the training activations (plm_sae.training.screen_fvu).
# The ProGen2-OAS TopK and Batch TopK screens ran at expansion 4 (dictionary size 6,144); the full runs use 8.
#
# Usage: scripts/training/lr_screen.sh train NAME LR    one screen run (one GPU, about 1 hour)
#        scripts/training/lr_screen.sh rank NAME        table of the five screen SAEs (one GPU, minutes)
# Output: $PLM_SAE_WORK_DIR/training/lr_screens/NAME/lr<LR>/NAME.safetensors and .../NAME/screen_fvu.tsv
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
source "$(dirname "${BASH_SOURCE[0]}")/sae_runs.sh"
MODE=$1; sae_run "$2" || exit 1
if [ "$NAME" = progen2oas_topk_r8 ] || [ "$NAME" = progen2oas_batchtopk_r8 ]; then EXP=4; sae_args; fi
ROOT="$PLM_SAE_WORK_DIR/training/lr_screens/$NAME"
case "$MODE" in
  train)
    LR=$3; OUT="$ROOT/lr$LR"
    [ -f "$OUT/$NAME.safetensors" ] && { echo "screen $NAME lr $LR done: $OUT"; exit 0; }
    mkdir -p "$OUT"
    log "screen $NAME lr $LR: $ARCH expansion $EXP -> $OUT"
    "$PY" -m plm_sae.training.train "${ARGS[@]}" +steps=10000 save_steps=10000 \
        sae_trainer_cfg.lr="$LR" save_dir="$OUT" hydra.run.dir="$OUT/hydra" || exit 1 ;;
  rank)
    printf 'lr\tresult\n' > "$ROOT/screen_fvu.tsv"
    for LR in 1e-7 1e-6 1e-5 1e-4 1e-3; do
      SAE="$ROOT/lr$LR/$NAME.safetensors"
      [ -f "$SAE" ] || { echo "missing $SAE"; continue; }
      R=$("$PY" -m plm_sae.training.screen_fvu "$SAE" "$MODEL" --n_seqs 128 | tail -n 1) || exit 1
      printf '%s\t%s\n' "$LR" "$R" | tee -a "$ROOT/screen_fvu.tsv"
    done ;;
  *) echo "usage: lr_screen.sh train NAME LR | rank NAME" >&2; exit 2 ;;
esac
