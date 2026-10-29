#!/bin/bash
# FlashTAP scores of the developability runs (paper.analyses.properties.outsel.tap, GPU, $PLM_SAE_PYTHON_FLASHTAP), an
# array of 8 shards over tap/runs.lst (the developability runs of manifest.csv). Run after every task of those runs is
# generated and numbered (unfinished runs are skipped; submit again to complete).
# Usage: paper/scripts/properties/submit_tap.sh
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
S="$PLM_SAE_WORK_DIR/outsel_properties"; mkdir -p "$S/tap"
"$PY" -c "import pandas as pd, sys; M = pd.read_csv(sys.argv[1] + '/manifest.csv'); open(sys.argv[1] + '/tap/runs.lst', 'w').write(''.join(f'{r}\n' for r in sorted(set(M[M.kind == 'dev'].run))))" "$S"
bash "$PLM_SAE_REPO/examples/slurm/submit.sh" "$PLM_SAE_REPO/paper/slurm/properties/tap.slurm" --array=0-7 --export=ALL,NSHARDS=8
