#!/bin/bash
# One ladder round of the property runs: the extension and refinement decision for every run
# (paper.analyses.properties.outsel.ladder PHASE, here), then submit_round.sh PHASE for the new tasks. Run after the
# previous round's ANARCI finished; repeat with new phase names (r1, r2, ...) until ladder_<phase>.csv reports every run
# FINAL.
# Usage: paper/scripts/properties/round.sh PHASE
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
S="$PLM_SAE_WORK_DIR/outsel_properties"; P=$1
[ -e "$S/tasks/$P.lst" ] && { echo "$P.lst exists; pick a new phase name"; exit 1; }
NPROC=${NPROC:-16} "$PY" -m paper.analyses.properties.outsel.ladder "$P" || exit 1
bash "$PLM_SAE_REPO/paper/scripts/properties/submit_round.sh" "$P"
