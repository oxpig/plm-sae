#!/bin/bash
# Binder status (Supplementary Section A.11): logistic-regression probe over the whole dictionary and the per-latent F1
# gate on the binder label, for the hyperparameter-matched p-IgGen TopK and Ordered SAEs (paper.analyses.binder.probe).
# CPU: 16 cores, 64 GB, about 6 min per SAE. Idempotent.
# Needs the per-token validation latents of both SAEs (scripts/extraction/encode.sh SAE val, as for the gates) and
# <data_dir>/datasets/rbd_binding_master.csv.
# Usage: paper/scripts/binder/probe.sh
# Writes <work_dir>/binder/<sae>/, then prints the A.11 numbers.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
set -eo pipefail
for SAE in piggen_topk_r8 piggen_ordered_r8; do
  ACT="$PLM_SAE_ACTIVATIONS/$SAE/val"
  [ -d "$ACT" ] || { echo "ERROR: missing $ACT; run scripts/extraction/encode.sh $SAE val" >&2; exit 1; }
  "$PY" -m paper.analyses.binder.probe run "$SAE" --ncpu "${SLURM_CPUS_PER_TASK:-16}"
done
"$PY" -m paper.analyses.binder.probe report
log "binder probe done"
