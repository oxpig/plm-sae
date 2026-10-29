#!/bin/bash
# Train the expansion-32 TopK SAEs of Section 4.1 (EleutherAI sparsify, all four p-IgGen block outputs in one run;
# plm_sae.training.sparsify_topk). One GPU. Needs the sparsify package (README).
#
# Usage: scripts/training/train_sparsify.sh [RUN_DIR]
# Input:  $PLM_SAE_DATA_DIR/datasets/oas_sae_training_dataset_unwrapped.csv (column `text`, VH+VL without 1 and 2)
# Output: RUN_DIR (default $PLM_SAE_WORK_DIR/training/sparsify_topk_r32/<start time>)/gpt_neox.layers.{0,1,2,3}/
#         sae.safetensors + cfg.json, and RUN_DIR/piggen_topk_r32_layer{0,1,2,3}.safetensors + .json, the format the
#         code reads: copy them into <data_dir>/checkpoints/ to use them by name
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
log "sparsify TopK r32 training"
"$PY" -m plm_sae.training.sparsify_topk ${1:+--run_name "$1"} || exit 1
log "done"
