#!/bin/bash
# Entry point of the paper's pipeline (paper/README.md gives the order, inputs and run times of every stage).
#
#   paper/reproduce.sh setup RECORD_DIR
#       unpack the data record (the files of the Zenodo record, downloaded to RECORD_DIR: datasets.tar,
#       checkpoints.tar, results.tar, libraries.tar) into <data_dir>: datasets/, checkpoints/, the unsteered reference
#       libraries and the result tables in <results_dir> (default <data_dir>/results). CPU, minutes; about 6 GB unpacked.
#   paper/reproduce.sh outputs [MANUSCRIPT_TEX]
#       tests, then every table and every figure from the result tables (CPU, a few minutes). With MANUSCRIPT_TEX,
#       Supplementary Table 11 and the binder numbers are also checked against the manuscript source.
#   paper/reproduce.sh submit PHASE
#       regenerate one phase from datasets and checkpoints as SLURM jobs (examples/slurm/submit.sh; site settings from
#       the environment, examples/slurm/README.md). Run the phases in this order, each after the previous one has
#       finished (or pass AFTER=<jobid>[:<jobid>...] to make the first jobs wait):
#         train        (optional) base-model activation dumps and the hyperparameter-matched SAEs
#         extract      latents of every SAE, IGHJ4 token latents
#         identify     gates, ranking lists, reconstruction, Section 4.1 analyses, IMGT localisation
#         benchmark    prints the steps of the steering benchmark (paper/README.md), which are run one after another
#         quality      generation quality of every library (one GPU job)
#         properties   developability gate and binder probe
#         ablation     latent ablation (GPU array) and dictionary redundancy
#         figures      GPU encode of Supplementary Figure 6
#       To build the tables from a regenerated run, set PLM_SAE_RESULTS_DIR to <work_dir>.
#       PLM_SAE_DRY_RUN=1 prints the sbatch commands without submitting.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"
usage() { sed -n '2,24p' "$0"; exit 2; }
[ $# -ge 1 ] || usage

# submit JOB [sbatch args...] -> prints the job id (dependencies: global DEP)
submit() {
  local job=$1; shift; local dep=()
  [ -n "${DEP:-}" ] && dep=(--dependency="afterok:$DEP")
  local out; out=$(bash examples/slurm/submit.sh "$job" --parsable "${dep[@]}" "$@")
  echo "${out%%;*}"
}
join() { local IFS=:; echo "$*"; }
# short CPU steps run here; printed only in a dry run
local_step() { if [ "${PLM_SAE_DRY_RUN:-0}" = 1 ]; then echo "local: $*" >&2; else "$@"; fi; }

case "$1" in
setup)
  [ $# -eq 2 ] || usage
  REC=$(cd "$2" && pwd); D="$PLM_SAE_DATA_DIR"; R="$PLM_SAE_RESULTS_DIR"
  for a in datasets checkpoints results; do   # extract the record's archives next to them, once
    [ -d "$REC/$a" ] || tar -xf "$REC/$a.tar" -C "$REC"
  done
  A0=libraries/libraries_piggen_alpha0_reference.tar.gz   # the only library archive setup needs
  [ -f "$REC/$A0" ] || tar -xf "$REC/libraries.tar" -C "$REC" "$A0"
  mkdir -p "$D/datasets" "$D/checkpoints" "$D/libraries" "$R"
  for f in "$REC"/datasets/*.csv.gz; do b=$(basename "$f" .gz); [ -e "$D/datasets/$b" ] || gunzip -c "$f" > "$D/datasets/$b"; done
  for f in "$REC"/checkpoints/*; do [ -e "$D/checkpoints/$(basename "$f")" ] || cp "$f" "$D/checkpoints/"; done
  for f in "$REC"/results/results_*.tar.gz; do tar -xzf "$f" -C "$R"; done
  tar -xzf "$REC/$A0" -C "$D/libraries"
  log "setup done: datasets, checkpoints and reference libraries in $D, result tables in $R" ;;

outputs)
  "$PY" tests/run_tests.py
  MANUSCRIPT="${2:-}" bash paper/scripts/figures/make_tables.sh
  bash paper/scripts/figures/make_figures.sh
  log "tables in $PLM_SAE_WORK_DIR/tables, figures in $PLM_SAE_WORK_DIR/figures" ;;

submit)
  [ $# -eq 2 ] || usage
  DEP=${AFTER:-}
  case "$2" in
  train)
    J=(); for m in piggen IgLM progen2_oas; do J+=($(submit examples/slurm/extraction/dump_activations.slurm --export=ALL,MODEL=$m)); done
    DEP=$(join "${J[@]}"); T=$(submit examples/slurm/training/train_sae.slurm)
    log "train: dumps ${J[*]}, SAEs $T (learning-rate screen and the sparsify SAEs: paper/README.md)" ;;
  extract)
    A=$(submit examples/slurm/extraction/encode.slurm)
    B=$(submit examples/slurm/extraction/encode.slurm --array=0-5 --time=03:00:00 --export=ALL,VARIANT=vhvl)
    C=$(submit paper/slurm/identification/section41_encode.slurm)
    E=$(submit paper/slurm/figures/encode_ighj4_tokens.slurm)
    log "extract: matched SAEs $A, p-IgGen VH+VL $B, Section 4.1 shards $C, IGHJ4 tokens $E" ;;
  identify)
    G=$(submit examples/slurm/identification/gate.slurm)
    GS=$(submit paper/slurm/identification/section41_gate.slurm --array=3)
    R=$(submit examples/slurm/identification/reconstruction.slurm)
    BASE=$DEP; DEP=$(join ${BASE:+$BASE} "$GS"); SA=$(submit paper/slurm/identification/section41_analyses.slurm)
    DEP=$(join ${BASE:+$BASE} "$G" "$GS"); PL=$(submit examples/slurm/run_script.slurm --export=ALL,SCRIPT=paper/scripts/identification/make_pair_lists.sh)
    DEP=$PL; IM=$(submit paper/slurm/identification/imgt.slurm)
    log "identify: gates $G, Section 4.1 gate $GS (layers 0-2: --array=0-2), reconstruction $R, Section 4.1 analyses $SA, ranking lists $PL, IMGT $IM" ;;
  benchmark)
    sed -n '/^## Steering benchmark/,/^## Generation quality/p' paper/README.md
    log "benchmark: run the steps above (GPU steps: paper/slurm/benchmark/gpu_step.slurm, loops: paper/slurm/benchmark/controller.slurm)" ;;
  quality)
    X=$(submit examples/slurm/run_script_gpu.slurm --export=ALL,SCRIPT=paper/scripts/quality/run_all.sh)
    log "quality: $X (needs the benchmark and progen2_t10 results)" ;;
  properties)
    F=$(submit paper/slurm/properties/flashtap_val.slurm)
    BASE=$DEP; DEP=$(join ${BASE:+$BASE} "$F"); DG=$(submit paper/slurm/properties/dev_gate.slurm)
    DEP=$BASE; BP=$(submit paper/slurm/properties/binder_probe.slurm)
    log "properties: FlashTAP $F, developability gate $DG, binder probe $BP; property steering: paper/README.md" ;;
  ablation)
    local_step "$PY" -m paper.analyses.ablation.make_tasks
    AB=$(submit paper/slurm/ablation/ablate.slurm); RD=$(submit paper/slurm/ablation/redundancy.slurm)
    log "ablation: $AB (then python -m paper.analyses.ablation.score), redundancy $RD" ;;
  figures)
    S6=$(submit paper/slurm/figures/ighj4_positions.slurm)
    log "figures: Supplementary Figure 6 encode $S6; then paper/reproduce.sh outputs" ;;
  *) usage ;;
  esac ;;
*) usage ;;
esac
