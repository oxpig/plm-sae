#!/bin/bash
# Feature identification gate of one hyperparameter-matched SAE (Methods; Supplementary Tables 4-6, the maximum F1 of
# Section 4.3, the light-chain-type gate of Supplementary Section A.7). plm_sae.features.gate, every latent scored.
#   scripts/identification/run_gate.sh NAME [j|lct]
#     j   (default) J-gene gate: labels j_gene_anarci_heavy (the ANARCI heavy-chain J call, allele removed) of
#         vj_{train,val}_anarci.csv, 10,000 training sequences
#     lct light-chain-type gate (p-IgGen, paired VH+VL input): labels light_chain_type of
#         datasets/vj_{train,val}_anarci_lct.csv, 6,000 training sequences
# Reads   <work_dir>/activations/NAME/{train,val}[_vhvl]/   (scripts/extraction/encode.sh)
# Writes  <work_dir>/identification/gates/NAME[_lct]/{best_model_pipeline.pkl,weights,f1_scores,margin_sweep}
# CPU only: 16 cores, 64 GB, up to 8 h per gate.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
NAME=${1:?usage: run_gate.sh NAME [j|lct]}
case "${2:-j}" in
  j)   SUB=""; OUT="$PLM_SAE_WORK_DIR/identification/gates/$NAME"; COL=j_gene_anarci_heavy; N=10000
       TRAIN_CSV="$PLM_SAE_DATA_DIR/datasets/vj_train_anarci.csv"; VAL_CSV="$PLM_SAE_DATA_DIR/datasets/vj_val_anarci.csv" ;;
  lct) SUB="_vhvl"; OUT="$PLM_SAE_WORK_DIR/identification/gates/${NAME}_lct"; COL=light_chain_type; N=6000
       TRAIN_CSV="$PLM_SAE_DATA_DIR/datasets/vj_train_anarci_lct.csv"
       VAL_CSV="$PLM_SAE_DATA_DIR/datasets/vj_val_anarci_lct.csv" ;;
  *) echo "label must be j or lct" >&2; exit 2 ;;
esac
if [ -f "$OUT/margin_sweep/summary.csv" ]; then log "done already: $OUT"; exit 0; fi
TOPK=$(( $("$PY" -c "from plm_sae import sae; print(sae.info('$NAME')['dictionary_size'])") + 1 ))   # every latent
log "gate $NAME (${2:-j}) -> $OUT"
"$PY" -m plm_sae.features.gate "$PLM_SAE_ACTIVATIONS/$NAME/train$SUB" "$TRAIN_CSV" "$PLM_SAE_ACTIVATIONS/$NAME/val$SUB" \
  "$VAL_CSV" "$COL" "$OUT" --top_k "$TOPK" --sample_n "$N" --sample_seed 42 --cpus "${SLURM_CPUS_PER_TASK:-16}"
cat "$OUT/margin_sweep/summary.csv"
