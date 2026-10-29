#!/bin/bash
# One ablation task (line IDX+1 of <work_dir>/ablation/tasks.txt, written by `python -m paper.analyses.ablation.make_tasks`):
# 1,000 p-IgGen sequences with the latent(s) ablated at the final-layer-norm output (strength 1) and the control
# (strength 0), seed 0, then ANARCI. One GPU, about 1 h. Skips the task if the ablated library's ANARCI table exists.
# Output: <work_dir>/ablation/piggen/ablation_<single|group>/
# Usage: paper/scripts/ablation/ablate.sh IDX          (IDX 0-17 single latents, 18-39 co-activation groups)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -euo pipefail
IDX=${1:?usage: ablate.sh IDX}
L=$(sed -n "$((IDX + 1))p" "$PLM_SAE_WORK_DIR/ablation/tasks.txt")
[ -n "$L" ] || { echo "no task $IDX"; exit 0; }
read -r kind sae gene lats <<< "$L"
OUT="$PLM_SAE_WORK_DIR/ablation/piggen/ablation_$kind"
DONE="$OUT/VH_alignment_layer_3_gene_${gene}_features_[${lats//+/, }]_alpha_1.00_seed_0_sequences_1000_formatted.csv"
[ -f "$DONE" ] && { echo "task $IDX already done"; exit 0; }
mkdir -p "$OUT"
echo "$(date '+%T') START $kind $sae $gene $lats"
# shellcheck disable=SC2086
"$PY" -m paper.analyses.ablation.ablate --sae "$sae" --latents ${lats//+/ } --gene "$gene" --output_dir "$OUT"
[ -f "$DONE" ] || { echo "FAILED task $IDX"; exit 1; }
echo "$(date '+%T') DONE task $IDX"
