# Training settings of every hyperparameter-matched SAE of the paper (Methods, Supplementary Tables 4 and 5).
# Sourced by train_sae.sh and lr_screen.sh. `sae_run NAME` sets
#   MODEL   base model (piggen | IgLM | progen2_oas)
#   ARCH    sae_trainer_cfg group: top_k | batch_top_k | ordered (configs/training/sae_trainer_cfg/)
#   EXP     expansion factor (dictionary size = d_model * EXP)
#   LR      learning rate chosen by the screens
#   ARGS    the Hydra overrides shared by the full runs and the screens (not steps, learning rate or save_dir), built
#           by sae_args (call it again after changing EXP)
# Every run: k 32, warmup 100 steps, seed 0, auxiliary-loss alpha 1/32, one sequence per step, activations divided by
# the norm factor of the training data (computed at the start of the run; plm_sae.models.NORM_FACTOR).
SAE_RUNS="piggen_topk_r8 piggen_batchtopk_r8 piggen_ordered_r8 iglm_topk_r8 iglm_batchtopk_r8 iglm_ordered_r8 \
progen2oas_topk_r8 progen2oas_batchtopk_r8 progen2oas_ordered_r8 piggen_topk_r32_matched"

sae_run() {
  case "$1" in
    piggen_topk_r8)          MODEL=piggen;      ARCH=top_k;       EXP=8;  LR=1e-4 ;;
    piggen_batchtopk_r8)     MODEL=piggen;      ARCH=batch_top_k; EXP=8;  LR=1e-4 ;;
    piggen_ordered_r8)       MODEL=piggen;      ARCH=ordered;     EXP=8;  LR=1e-4 ;;
    iglm_topk_r8)            MODEL=IgLM;        ARCH=top_k;       EXP=8;  LR=1e-4 ;;
    iglm_batchtopk_r8)       MODEL=IgLM;        ARCH=batch_top_k; EXP=8;  LR=1e-4 ;;
    iglm_ordered_r8)         MODEL=IgLM;        ARCH=ordered;     EXP=8;  LR=1e-3 ;;
    progen2oas_topk_r8)      MODEL=progen2_oas; ARCH=top_k;       EXP=8;  LR=1e-4 ;;
    progen2oas_batchtopk_r8) MODEL=progen2_oas; ARCH=batch_top_k; EXP=8;  LR=1e-4 ;;
    progen2oas_ordered_r8)   MODEL=progen2_oas; ARCH=ordered;     EXP=8;  LR=1e-5 ;;
    piggen_topk_r32_matched) MODEL=piggen;      ARCH=top_k;       EXP=32; LR=1e-4 ;;
    *) echo "unknown SAE '$1'; one of: $SAE_RUNS" >&2; return 1 ;;
  esac
  NAME=$1; sae_args
}

sae_args() {
  local dm
  case "$MODEL" in piggen) dm=piggen ;; IgLM) dm=iglm ;; progen2_oas) dm=progen2_oas ;; esac
  ARGS=(datamodule="$dm" sae_trainer_cfg="$ARCH" expansion_factor="$EXP" sae_trainer_cfg.k=32
        sae_trainer_cfg.warmup_steps=100 log_steps=1000 seed=0 name="$NAME")
}
