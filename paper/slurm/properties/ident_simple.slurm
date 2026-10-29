#!/bin/bash
# Sequence-level F1 of the p-IgGen latents for the CDRH3 properties (paper.analyses.properties.ident_simple; CPU).
#   examples/slurm/submit.sh paper/slurm/properties/ident_simple.slurm
#SBATCH --job-name=properties-ident
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
set -euo pipefail
source "$PLM_SAE_REPO/scripts/env.sh"
"$PY" -u -m paper.analyses.properties.ident_simple
