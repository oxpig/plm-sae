#!/bin/bash
# Deduplicated ANARCI numbering (IMGT scheme, germline assignment) of the libraries of a units file, spread over SLURM
# CPU jobs: the unique sequences are collected here (plm_sae.numbering.number collect), numbered by an array of NT
# jobs with C CPUs each, and a dependent assemble job writes each library's VH_alignment and _formatted CSVs next to it.
# Completion marker: HI_DONE as the last line of WORKDIR/assemble.log and the file WORKDIR/HI_DONE. Libraries whose
# formatted CSV already exists are skipped. Single-node equivalent: scripts/numbering/number_units.sh.
#
# Usage: paper/scripts/benchmark/anarci_array.sh UNITS WORKDIR TAG [NT=3] [C=32]      (UNITS lines: "dir latent gene alpha seed")
# Jobs are named anarci_<TAG>_num and anarci_<TAG>_asm (paper/scripts/benchmark/common.sh uses TAG = first 14 characters of the
# task name + "_" + 4 hex digits; benchmark/extend.py looks for that prefix). Prints "number <jobid> assemble <jobid>".
# Environment (defaults are the paper's settings): ANARCI_NUM_MEM (60G), ANARCI_ASM_MEM (120G), ANARCI_NUM_TIME (2:00:00),
# ANARCI_ASM_TIME (1:00:00), ANARCI_PARTITION (default PLM_SAE_SLURM_CPU_PARTITION), ANARCI_SBATCH (extra sbatch flags),
# PLM_SAE_SLURM_CLUSTER (sbatch --clusters, if set).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
[ $# -ge 3 ] || { echo "usage: $0 UNITS WORKDIR TAG [NT] [C]"; exit 1; }
mkdir -p "$2"; U=$(realpath "$1"); W=$(realpath "$2"); TAG=$3; NT=${4:-3}; C=${5:-32}
rm -f "$W"/res_*.pkl "$W/HI_DONE"
"$PY" -m plm_sae.numbering.number collect "$U" "$W" | tee "$W/collect.log"
PART=${ANARCI_PARTITION:-$PLM_SAE_SLURM_CPU_PARTITION}
SB=(sbatch --parsable --export=ALL ${PLM_SAE_SLURM_CLUSTER:+--clusters=$PLM_SAE_SLURM_CLUSTER} ${PART:+--partition=$PART})
# job scripts: hmmscan from the environment's bin directory, single-threaded (the parallelism is over sequences)
cat > "$W/number_job.sh" <<EOF
#!/bin/bash
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
export PATH="\$(dirname "\$PY"):\$PATH" HMMER_NCPU=1
"\$PY" -m plm_sae.numbering.number number "$W" "\$SLURM_ARRAY_TASK_ID" $NT $C
EOF
cat > "$W/assemble_job.sh" <<EOF
#!/bin/bash
source "$PLM_SAE_REPO/scripts/env.sh" || exit 1
export PATH="\$(dirname "\$PY"):\$PATH" HMMER_NCPU=1
"\$PY" -m plm_sae.numbering.number assemble "$U" "$W" $C && { echo HI_DONE; touch "$W/HI_DONE"; }
EOF
A=$("${SB[@]}" --job-name="anarci_${TAG}_num" $ANARCI_SBATCH --cpus-per-task=$C --mem=${ANARCI_NUM_MEM:-60G} \
     --time=${ANARCI_NUM_TIME:-2:00:00} --array=0-$((NT-1)) --output="$W/number_%a.log" "$W/number_job.sh" | cut -d';' -f1)
[ -n "$A" ] || { echo "number job not submitted"; exit 1; }
B=$("${SB[@]}" --job-name="anarci_${TAG}_asm" $ANARCI_SBATCH --cpus-per-task=$C --mem=${ANARCI_ASM_MEM:-120G} \
     --time=${ANARCI_ASM_TIME:-1:00:00} --dependency=afterok:$A --output="$W/assemble.log" "$W/assemble_job.sh" | cut -d';' -f1)
echo "number $A assemble $B"
