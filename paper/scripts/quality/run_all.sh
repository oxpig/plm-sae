#!/bin/bash
# Steered generation analysis (Table 3, Supplementary Tables 10 and 11, the library medians of Sections 4.2.3 and 4.3),
# every step in order on one machine with a GPU; each step skips finished work. On a cluster, run steps 2, 4 and 6
# (CPU, seconds) directly and submit paper/slurm/quality/natural.slurm, libraries.slurm and ablang.slurm for 1, 3 and 5.
#   1 natural references   python -m paper.analyses.quality.natural
#   2 library list         python -m paper.analyses.quality.libraries        (needs the benchmark and progen2_t10 outputs)
#   3 per library          python -m paper.analyses.quality.run_one ROW     (rows 0-2, the unsteered references, first)
#   4 collect              python -m paper.analyses.quality.summarise
#   5 AbLang-2             "$PLM_SAE_PYTHON_ABLANG" -m paper.analyses.quality.ablang_scores run --fp16
#   6 AbLang-2 tests       python -m paper.analyses.quality.ablang_summary
# Usage: paper/scripts/quality/run_all.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
Q="$PLM_SAE_WORK_DIR/quality"
[ -f "$Q/natural_heavy.npz" ] || "$PY" -m paper.analyses.quality.natural
"$PY" -m paper.analyses.quality.libraries
n=$(( $(wc -l < "$Q/list.csv") - 1 ))
for i in $(seq 0 $((n - 1))); do "$PY" -m paper.analyses.quality.run_one "$i"; done
"$PY" -m paper.analyses.quality.summarise
"$PLM_SAE_PYTHON_ABLANG" -m paper.analyses.quality.ablang_scores run --fp16
"$PY" -m paper.analyses.quality.ablang_summary
