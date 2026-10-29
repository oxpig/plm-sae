#!/bin/bash
# Train one hyperparameter-matched SAE of the paper: 100,000 steps on the activation cache of its base model
# (python -m plm_sae.activations.dump), settings from sae_runs.sh. One GPU; 6 to 14 hours.
#
# Usage: scripts/training/train_sae.sh NAME        NAME: an SAE of the data record, e.g. piggen_topk_r8 (sae_runs.sh)
# Output: $PLM_SAE_WORK_DIR/training/NAME/NAME.safetensors + NAME.json (the record's format) and sae-step-100000.ckpt.
# Skips if the export exists. To use a retrained SAE by name, copy NAME.safetensors and NAME.json into
# <data_dir>/checkpoints/ (in place of the record's copies).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
source "$(dirname "${BASH_SOURCE[0]}")/sae_runs.sh"
sae_run "$1" || exit 1
OUT="$PLM_SAE_WORK_DIR/training/$NAME"
[ -f "$OUT/$NAME.safetensors" ] && { echo "$NAME already trained: $OUT"; exit 0; }
mkdir -p "$OUT"
log "train $NAME: $MODEL $ARCH expansion $EXP lr $LR -> $OUT"
"$PY" -m plm_sae.training.train "${ARGS[@]}" +steps=100000 save_steps=100000 \
    sae_trainer_cfg.lr="$LR" save_dir="$OUT" hydra.run.dir="$OUT/hydra" || exit 1
log "done $NAME"
