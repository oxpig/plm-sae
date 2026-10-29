#!/bin/bash
# Every table body and every number of the text that is computed from result tables (CPU, seconds each). Reads each
# stage's results from <work_dir> where present, else <results_dir> (the data record). Writes <work_dir>/tables and
# prints every row; paper/CHECKLIST.md maps each table and number to its step.
#   Table 2 (tab:benchmark)                         python -m paper.tables.table2
#   Table 3, Supplementary Table 10, text medians   python -m paper.tables.table3_rows
#   Supplementary Table 7 (tab:rl_perpair)          python -m paper.tables.rl_perpair
#   Supplementary Table 8 (tab:rl_depth)            python -m paper.tables.rl_depth
#   Supplementary Table 9 (tab:gate)                python -m paper.tables.gate
#   Supplementary Table 11 (tab:t3_batch)           python -m paper.analyses.quality.by_set check --tex $MANUSCRIPT
#   Supplementary Table 15 (tab:lct_steering)       python -m paper.tables.lct_steering
#   Supplementary Table 19 (tab:dev_counts)         python -m paper.tables.dev_counts
#   Supplementary Tables 20-22                      python -m paper.tables.outsel_properties
#   Section 4.2.1, expansion-32 TopK steering       python -m paper.tables.section41_steering
#   Sections 4.2.3 and 4.3, Mantel-Haenszel tests   python -m paper.tables.benchmark_tests
#   Supplementary A.4, natural scale                python -m paper.benchmark.natural_scale compare
#   Supplementary A.4, CDRH3 homopolymers           python -m paper.benchmark.homopolymer summary
#   Supplementary A.4, stricter readout             python -m paper.benchmark.strict summary
#   Supplementary A.8, joint steering               python -m paper.tables.joint_steering
#   Supplementary A.11, binder status               python -m paper.analyses.binder.probe report
#   Output selection reproduces the steered sets    python -m paper.benchmark.check_picks
# Usage: [MANUSCRIPT=path/manuscript.tex] paper/scripts/figures/make_tables.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
TEX=()
[ -n "${MANUSCRIPT:-}" ] && TEX=(--tex "$MANUSCRIPT")
"$PY" -m paper.tables.table2 --out "$PLM_SAE_WORK_DIR/tables"
"$PY" -m paper.tables.table3_rows
"$PY" -m paper.tables.rl_perpair
"$PY" -m paper.tables.rl_depth
"$PY" -m paper.tables.gate
[ -n "${MANUSCRIPT:-}" ] && "$PY" -m paper.analyses.quality.by_set check "${TEX[@]}"
"$PY" -m paper.tables.lct_steering
"$PY" -m paper.tables.dev_counts
"$PY" -m paper.tables.outsel_properties
"$PY" -m paper.tables.section41_steering
"$PY" -m paper.tables.benchmark_tests
"$PY" -m paper.benchmark.natural_scale compare
"$PY" -m paper.benchmark.homopolymer summary
"$PY" -m paper.benchmark.strict summary
"$PY" -m paper.tables.joint_steering
"$PY" -m paper.analyses.binder.probe report "${TEX[@]}"
"$PY" -m paper.benchmark.check_picks
