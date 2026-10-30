# Reproducing the paper

This folder reproduces the tables, figures and numbers of the paper. It uses the pipeline code in `plm_sae/` and
adds the steering benchmark, the supplementary analyses and the table and figure code.

## Getting started
1. Install the environment (see the top-level `README.md`). The AbLang-2 and FlashTAP analyses run in their own
   environments:
   ```sh
   conda env create -f envs/ablang2.yml     # then set python_ablang in configs/paths.yaml to its python
   conda env create -f envs/flashtap.yml    # then set python_flashtap in configs/paths.yaml to its python
   ```
2. Download the files of the Zenodo record (https://doi.org/10.5281/zenodo.23813269) into a folder `<record_dir>`:
   `datasets.tar`, `checkpoints.tar`, `results.tar` and `libraries.tar` (the last is needed only for its unsteered
   reference libraries and to rescore steered libraries).
3. Unpack the datasets, checkpoints, reference libraries and result tables into `<data_dir>` (CPU, a few minutes,
   about 6 GB):
   - `bash paper/reproduce.sh setup <record_dir>`
4. Run the tests and regenerate every table and figure from the result tables (CPU, a few minutes):
   - `bash paper/reproduce.sh outputs`
   - Note: add the path to `manuscript.tex` to also check Supplementary Table 11 and the binder numbers against the text.

To regenerate the results from the datasets and checkpoints (GPU, SLURM), run one phase at a time:
- `bash paper/reproduce.sh submit <phase>`
- Phases in order: `train`, `extract`, `identify`, `benchmark`, `quality`, `properties`, `ablation`, `figures`.
- The steps of each phase are below. To build the tables from a regenerated run, set `PLM_SAE_RESULTS_DIR` to
  `<work_dir>`.

Every stage reads its inputs from `<work_dir>` when an earlier stage wrote them there, and otherwise from the record
(`<results_dir>`), so a stage can be rerun on its own.

## Layout
- `reproduce.sh`: the entry point (`setup`, `outputs`, `submit`).
- `benchmark/`: the steering benchmark: latent sets, task files, strength ladders, scoring, the ProGen2-OAS run at
  temperature 1.0 (`progen2_t10/`), windowed steering (`windowed/`) and the follow-up analyses.
- `analyses/`: identification (Section 4.1 gate, IMGT localisation), generation quality, property steering,
  ablation, redundancy and binder status.
- `tables/`, `figures/`: table and figure code (CPU).
- `scripts/`, `slurm/`: shell scripts and SLURM jobs of the steps above.
- `CHECKLIST.md`: every table and figure, its command and its input in the record.

## Data record
The record holds four archives, each extracting to one folder, plus the code snapshot, a README and checksums.
- `datasets/`: SAE training sets, train and validation splits with ANARCI labels, natural references, the binder table.
- `checkpoints/`: every SAE (`<name>.safetensors` and `<name>.json`).
- `results/`: one archive per stage with the result tables behind every table, figure and number, in the layout the
  code writes (`<stage>/...`).
- `libraries/`: the steered libraries and their ANARCI tables, with paths relative to `<work_dir>` (unpack an archive
  into `<work_dir>` to rescore it). `libraries_piggen_alpha0_reference` goes into `<data_dir>/libraries/`.

Activation caches (about 660 GB) are not in the record. The extraction step regenerates them.

Four inputs come only with the record (no step produces them): `redundancy/coactivation_groups.txt` (the 22
co-activation groups), `ablation/single_latents.txt` (the ablated latents), `identification/lowidx/selection.csv`
(Ordered latents excluded from the baseline draw) and `outsel_properties/baseline_exclusions.csv` (latents excluded
from the property baselines).

## Training
1. `bash scripts/extraction/dump_activations.sh <model>` for `piggen`, `IgLM`, `progen2_oas` (GPU).
2. `bash scripts/training/train_sae.sh <sae>` for the nine SAEs `<model>_<topk|batchtopk|ordered>_r8` and
   `piggen_topk_r32_matched` (GPU, `scripts/training/sae_runs.sh` lists them). Learning rates were picked with
   `scripts/training/lr_screen.sh`.
3. `bash scripts/training/train_sparsify.sh`: the expansion-32 TopK SAEs of Section 4.1 (p-IgGen layers 0-3).

## Extraction and identification
1. `bash scripts/extraction/encode.sh <sae> train|val` for the nine SAEs, and `... <p-IgGen sae> train|val vhvl` for
   the light-chain-type gate (GPU).
2. `bash scripts/identification/run_gate.sh <sae> [j|lct]` (CPU, up to 8 h): J-gene gates (Supplementary Tables 4-6)
   and light-chain-type gates (Supplementary Section A.7). The J-gene gates label each sequence with its ANARCI
   heavy-chain J call (`j_gene_anarci_heavy`, allele removed), so the latent ranking, the pair lists of step 3 and
   everything the benchmark steers follow the ANARCI call.
3. `bash paper/scripts/identification/make_pair_lists.sh`: the ranked latents the benchmark steers.
4. `bash scripts/identification/run_reconstruction.sh <model>` (GPU): FVU, L0 and dead latents (Supplementary
   Tables 4 and 5) and the firing counts used by the developability counts and the redundancy analysis.
5. Section 4.1: `paper/scripts/identification/section41_encode.sh <task> [layer]` (32 shards, GPU),
   `section41_gate.sh [layer]` (Table 1, Supplementary Tables 12 and 13), and `section41_analyses.sh cdr|residue|stats`
   (CDR identity probe, IMGT positions of latents 463, 4720 and 6276, the statistics in the Supplementary Table 4
   caption).
6. `bash paper/scripts/figures/encode_ighj4_tokens.sh <iglm_topk_r8|progen2oas_topk_r8>` (GPU), then
   `bash paper/scripts/identification/run_imgt.sh` (CPU): Supplementary Table 14 and Supplementary Figures 8 and 9.

## Steering benchmark
`B` is `<work_dir>/benchmark`. GPU steps take 10 to 60 minutes per model on one 24 GB GPU. Generation runs as task
files in `B/tasks/todo`, claimed by GPU workers (`paper/slurm/benchmark/gpu_worker.slurm`, one per GPU); the loops
below run as CPU jobs (`paper/slurm/benchmark/controller.slurm` with `LOOP=<name>`).
1. Start positions: `python -m paper.benchmark.positions <model>` for the three models and
   `python -m paper.benchmark.positions_generic` (GPU).
2. Baseline latents: `python -m paper.benchmark.baselines <model> --set 1` (GPU), then `--set 2` (CPU).
3. Output scores: `python -m plm_sae.steering.output_scores <model>` (GPU). The output-selected latents follow from
   them (`plm_sae.steering.pick_output`).
4. Tasks: `python -m paper.benchmark.make_tasks gated`, then `ungated`, `joint` and `other`.
5. Workers, and the loop `chain.sh`: it numbers every finished library with ANARCI, extends the ladder of latents
   whose coverage is still at least 80% at the largest strength (`paper.benchmark.extend`), and runs
   `paper.benchmark.analyze_final` (`B/final_latents.csv`, `B/final_libraries.csv.gz`) when nothing is left. Once the
   ladders are scored, `python -m paper.benchmark.make_refine` adds strengths for latents with fewer than five valid
   strengths; delete `B/ALL_TASKS_ADDED` and run the loop again.
6. ProGen2-OAS at temperature 1.0: `python -m paper.benchmark.progen2_t10.make_tasks`, workers with
   `QLIST="<work_dir>/progen2_t10:todo:2700"` on GPUs with TF32 (Ampere or newer), and the loops
   `progen2_t10_number.sh` and `progen2_t10_ladder.sh` (scores in `<work_dir>/progen2_t10/score/`).
7. Natural scale and windowed steering (Supplementary Section A.4): `python -m paper.benchmark.natural_scale measure
   <model>` (GPU), `python -m paper.benchmark.windowed.make_win`, workers with `QLIST="B/win:todo:2700"`, the loop
   `windowed_number.sh`, then `python -m paper.benchmark.windowed.analyze_win`.
8. Follow-up analyses: `python -m paper.benchmark.strict gated` and `existing` (stricter readout),
   `python -m paper.benchmark.homopolymer measure` (CDRH3 homopolymers), `python -m paper.benchmark.mediation <model>`
   (GPU, Supplementary Section A.3) and `python -m paper.benchmark.calib_alpha`.

## Generation quality
`bash paper/scripts/quality/run_all.sh` on one GPU node, or the jobs in `paper/slurm/quality/` in the order of its
header: natural references, the library list, the metrics of each library, then AbLang-2 (Table 3, Supplementary
Tables 10 and 11, Supplementary Figure 4).

## Properties
- Developability gate (Supplementary Section A.10, Supplementary Table 19): `paper/scripts/properties/flashtap_val.sh`
  (GPU), `developability_gate.sh <sae>` for the nine SAEs, then `python -m paper.analyses.properties.developability.counts`.
- Steering the developability latents 980 and 1088: their libraries come from the benchmark (`make_tasks other`);
  `dev_score.sh 0|1` scores them with FlashTAP, `python -m paper.analyses.properties.dev_steering.analysis trend`,
  `analysis controls`, `dev_ctrl.sh 0..5`, `analysis compare`.
- Output selection for CDRH3 properties and developability (Supplementary Section A.12, Supplementary Tables 20-22):
  `python -m paper.analyses.properties.ident_simple`, `python -m paper.analyses.properties.outsel.scores` (GPU),
  `python -m paper.analyses.properties.outsel.manifest`, `paper/scripts/properties/submit_round.sh base`, then
  `round.sh r1`, `round.sh r2`, ... until every run is final, `submit_tap.sh`, and
  `python -m paper.analyses.properties.outsel.analyze`. `python -m paper.analyses.properties.cdr3_check` checks the
  CDRH3 parser on the validation heavy chains.
- Lambda light chains: `python -m paper.analyses.properties.lambda_steering tasks`, workers, then `... score`.
- Binder status (Supplementary Section A.11): `paper/scripts/binder/probe.sh` (CPU).

## Ablation and redundancy
`python -m paper.analyses.ablation.make_tasks`, `paper/slurm/ablation/ablate.slurm` (GPU, 40 tasks), then
`python -m paper.analyses.ablation.score` (Supplementary Tables 17 and 18). `paper/scripts/redundancy/run.sh`:
dictionary redundancy (Supplementary Table 16). The 22 co-activation groups are an input of the record
(`redundancy/coactivation_groups.txt`).

## Tables and figures
`paper/scripts/figures/make_tables.sh` and `make_figures.sh` (CPU). `CHECKLIST.md` lists each table and figure with
its command and input.

## Tests
- `python tests/run_tests.py` checks that every module imports, rebuilds Table 2 and Supplementary Tables 7, 8 and
  19-22 from small fixtures, and reproduces the output-selected latents.
