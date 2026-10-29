#!/bin/bash
# Per-token latents of the IgLM or ProGen2-OAS TopK SAE for the IGHJ4 validation sequences (10,269 of 30,754), the input
# of the IMGT activation figure of the top IGHJ4 latents (Supplementary Figure 8; paper.analyses.identification.
# imgt_position_counts). One GPU; resumable.
#
# Usage: paper/scripts/figures/encode_ighj4_tokens.sh iglm_topk_r8 | progen2oas_topk_r8
# Input:  $PLM_SAE_WORK_DIR/identification/imgt/vj_val_anarci_ighj4.csv (written here if missing)
# Output: $PLM_SAE_ACTIVATIONS/NAME/val_ighj4_tok/<orig_row>.npy, [200, dictionary size] float32 each
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
NAME=$1
case "$NAME" in
  iglm_topk_r8) BS=64 ;;
  progen2oas_topk_r8) BS=32 ;;
  *) echo "NAME must be iglm_topk_r8 or progen2oas_topk_r8" >&2; exit 2 ;;
esac
CSV="$PLM_SAE_WORK_DIR/identification/imgt/vj_val_anarci_ighj4.csv"
[ -f "$CSV" ] || "$PY" -m paper.analyses.identification.imgt || exit 1
OUT="$PLM_SAE_ACTIVATIONS/$NAME/val_ighj4_tok"
log "encode $NAME IGHJ4 validation tokens -> $OUT"
"$PY" -m plm_sae.activations.encode "$NAME" "$CSV" "$OUT" --pool tokens --batch_size "$BS" || exit 1
log "done $NAME"
