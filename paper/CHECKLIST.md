# Reproduction checklist

Every numbered figure and table of the manuscript (BIOINF-2026-1349), the command that produces it, the
result file it reads, and whether it was reproduced from this repository and the data record.

**Data record.** https://doi.org/10.5281/zenodo.23813269. After `paper/reproduce.sh setup RECORD_DIR`, result files sit
under `<results_dir>` (`data/results/` by default) in the layout the code writes; `results/results_<stage>.tar.gz` holds
`<stage>/`. Datasets are `datasets/*.csv.gz`, SAE weights `checkpoints/<name>.safetensors`.

**One command for everything that runs from the record:** `paper/reproduce.sh outputs [manuscript.tex]` (tests, every
table, every figure; CPU, a few minutes). Regenerating the inputs: `paper/reproduce.sh submit PHASE` and `README.md`.

**Status**

* **Reproduced**: regenerated from the record with the command shown. Figures were compared with the
  manuscript's PNG files byte for byte (matplotlib 3.10.9, as pinned in `environment.yml`). For tables, every row the
  command writes was compared with the manuscript table.
* **Record values match**: the table is read from a deposited output of a GPU or multi-hour step. The values were
  compared with the manuscript; the step itself was not rerun. Its code is present.
* **Code present, not run**: the code and the deposited output are present, but they were not compared.

## Figures

| No. | Label | Content | Command | Input in the record | Status |
|---|---|---|---|---|---|
| 1 | `fig:neuron-vs-latent` | Latent 4280 against hidden neuron 45 along one IGHJ3 heavy chain (expansion-32 TopK SAE, layer 3) | `python -m paper.figures.fig_latent_profiles OUT` (CPU, seconds) | `checkpoints/piggen_topk_r32_layer3`; p-IgGen weights (Hugging Face); the sequence is in the script | Reproduced, byte-identical |
| 2 | `fig:Successful steering` | IGHJ6 steering in p-IgGen: (a) highest-F1 Batch TopK latent 733, (b) output-selected TopK latent 5141 | `paper/scripts/figures/make_steering_figures.sh OUT` | `benchmark/final_latents.csv`, `benchmark/final_libraries.csv.gz` | Reproduced, byte-identical |
| 3 | `fig:xmodel_steering` | IgLM Ordered latent 1214 (IGHJ5), ProGen2-OAS Batch TopK latent 7567 (IGHJ3) | same | as Figure 2, and `progen2_t10/score/` | Reproduced, byte-identical |
| S4 | `fig:quality_violins` | Generation quality of the passing libraries by set, three models | `python -m paper.analyses.quality.by_set pool`; `python -m paper.figures.fig_quality_violins OUT/supp_quality_violins.png` | `quality/per_latent.csv`, `quality/per_library.csv`, `quality/ablang/scores/natural_*.npy` | Reproduced; 178 of 2,764,916 pixels differ (at most 8 of 255 in one channel), because three libraries carry the metric values reported in Table 3 and Supplementary Table 10 |
| S5 | `fig:ighj3_latent` | IGHJ3 latents 3054, 532, 3098 along one IGHJ3 heavy chain | `python -m paper.figures.fig_latent_profiles OUT` | as Figure 1 | Reproduced, byte-identical |
| S6 | `fig:IMGT_IGHJ4_activations` | Token (a) and IMGT (b) positions of IGHJ4 latents 463, 4720, 6276 over 10,269 validation heavy chains | `STAGES=plot paper/scripts/figures/ighj4_positions.sh OUT` (a fresh encode: `STAGES="table encode plot"`, GPU) | `figures/ighj4_positions/` | Reproduced, byte-identical (from the deposited encode) |
| S7 | `fig:Unsuccessful steering` | Expansion-32 TopK latents 463, 4720, 6276 steered toward IGHJ4 | `paper/scripts/figures/make_steering_figures.sh OUT` | `benchmark/` (set `section41`) | Reproduced, byte-identical |
| S8 | `fig:IgLM_ad_progen2_activations` | IMGT positions of the top IgLM and ProGen2-OAS TopK IGHJ4 latents | `python -m paper.figures.fig_imgt OUT` | `identification/imgt/imgt_position_counts.json` | Reproduced, byte-identical |
| S9 | `fig:nested latent activations` | IMGT positions of the highest-F1 Ordered and TopK p-IgGen latents | same | `identification/imgt/mass_<sae>.npz` | Reproduced, byte-identical |

`paper/scripts/figures/make_figures.sh OUT` draws all of them (about a minute on one CPU).

## Tables

| No. | Label | Content | Command | Input in the record | Status |
|---|---|---|---|---|---|
| 1 | `tab:performance` | Precision, recall, F1 per IGHJ class, expansion-32 TopK SAE | `paper/scripts/identification/section41_encode.sh` (GPU), `section41_gate.sh 3` (CPU, 6 h) | `identification/gates/piggen_topk_r32_layer3/full_model_evaluation_j_gene_anarci_heavy.csv` | Record values match (21/21 cells) |
| 2 | `tab:benchmark` | Steering benchmark, latents passing per set and model | `python -m paper.tables.table2 --out D` | `benchmark/`, `progen2_t10/` | Reproduced (9/9 rows) |
| 3 | `tab:steered_generation_analysis` | Batch TopK latent 733 library against natural sequences | `python -m paper.tables.table3_rows` | `quality/per_latent.csv`, `quality/per_library.csv`, `quality/ablang/` | Reproduced (6/6 rows) |
| S4 | `tab:matched_training` | Matched SAEs, p-IgGen: learning rate, FVU, L0, dead latents, F1 range | `scripts/training/*`, `scripts/identification/run_reconstruction.sh` (GPU), `run_gate.sh` | `identification/reconstruction/recon_val_*.csv`, `identification/gates/` | Code present, not run. Caption statistics of the expansion-32 SAE: `identification/reconstruction/section41_stats.csv` (FVU 0.0088, L0 32.00, 32.98% never fired; `paper/scripts/identification/section41_analyses.sh stats`) |
| S5 | `tab:matched_training_xmodel` | As S4, IgLM and ProGen2-OAS | as S4 | as S4 | Code present, not run |
| S6 | `tab:margin_counts` | Latents above F1 margins, nine matched SAEs, J labels from the ANARCI heavy-chain J call | `scripts/identification/run_gate.sh <sae> j` (CPU, up to 8 h each) | `identification/gates/<sae>/margin_sweep/counts_by_margin.csv` | Code present, not run |
| S7 | `tab:rl_perpair` | Per-pair detail of the highest-F1 latents | `python -m paper.tables.rl_perpair` | `benchmark/`, `progen2_t10/` | Reproduced (54/54 rows) |
| S8 | `tab:rl_depth` | Passing latents among ranks 1, 1-2, 1-3 | `python -m paper.tables.rl_depth` | same | Reproduced (11/11 rows) |
| S9 | `tab:gate` | Steering with and without the position gate | `python -m paper.tables.gate` | `benchmark/final_latents.csv` | Reproduced (5/5 rows and the numbers of the paragraph) |
| S10 | `tab:t3_output` | Output-selected TopK latent 5141 library against natural sequences | `python -m paper.tables.table3_rows` | as Table 3 | Reproduced (6/6 rows) |
| S11 | `tab:t3_batch` | Quality of every passing library per model, SAE and set | `python -m paper.analyses.quality.by_set check --tex manuscript.tex` | `quality/` | Reproduced (462/462 cells) |
| S12 | `tab:IGHJ4_layer_performance` | IGHJ4 features and max F1, p-IgGen layers 0-3 | `paper/scripts/identification/section41_gate.sh <layer>` | `identification/gates/piggen_topk_r32_layer{0,1,2,3}/` | Record values match (features 2, 12, 13, 11; max F1 0.885, 0.960, 0.967, 0.878) |
| S13 | `tab:ighj-features` | Layer-3 feature counts and max F1 per gene | as Table 1 | `identification/gates/piggen_topk_r32_layer3/` | Record values match (counts 8, 13, 23, 11, 6, 26) |
| S14 | `tab:localisation` | J-region share of the highest-F1 latent per gene | `paper/scripts/identification/run_imgt.sh` (CPU, 40 processes) | `identification/imgt/localisation.csv` | Record values match |
| S15 | `tab:lct_steering` | Light-chain-type steering | `python -m paper.tables.lct_steering` | `benchmark/final_latents.csv` | Reproduced (18/18 rows) |
| S16 | `tab:splitting` | Dictionary redundancy, nine matched SAEs | `paper/scripts/redundancy/run.sh` | `redundancy/redundancy.csv` | Code present, not run |
| S17 | `tab:ablation_single` | Single-latent ablation | `python -m paper.analyses.ablation.make_tasks`, `paper/slurm/ablation/ablate.slurm` (GPU), `python -m paper.analyses.ablation.score` | `ablation/results/ablation.csv` | Code present, not run |
| S18 | `tab:ablation_cluster` | Co-activation group ablation | as S17 | as S17 | Code present, not run |
| S19 | `tab:dev_counts` | Developability latents per SAE | `python -m paper.tables.dev_counts` | `developability/counts/dev_latent_counts.csv` | Reproduced (9/9 rows) |
| S20 | `tab:dev_latents` | Developability steering, every latent | `python -m paper.tables.outsel_properties` | `outsel_properties/results_dev.csv` | Reproduced (24/24 rows) |
| S21 | `tab:graded` | Graded CDRH3 steering, passes per set | same | `outsel_properties/results_graded.csv` | Reproduced (9/9 rows) |
| S22 | `tab:graded_latents` | Graded latents | same | same | Reproduced (27/27 rows) |

`paper/scripts/figures/make_tables.sh` runs every "Reproduced" table command. With `MANUSCRIPT=manuscript.tex` it also
checks Supplementary Table 11 and the binder numbers against the manuscript.

## Numbers in the text

| Item | Command | Status |
|---|---|---|
| Section 4.2.1, expansion-32 TopK SAE: 6 of 18 latents, genes IGHJ1, IGHJ3, IGHJ4 | `python -m paper.tables.section41_steering` | Reproduced |
| Sections 4.2.3 and 4.3, Mantel-Haenszel p (0.52, 0.28, 0.68; output-selected < 0.001) | `python -m paper.tables.benchmark_tests` | Reproduced |
| Sections 4.2.3 and 4.3, library medians | `python -m paper.tables.table3_rows` | Reproduced |
| A.3 position gate paragraph (15 and 12 of 54, p = 0.45; TV 0.095 and 0.024, 0.087 and 0.023; IgLM 22 to 12, 39 to 21) | `python -m paper.tables.gate` | Reproduced |
| A.4 output-selected latents (54 J-gene, 12 developability, 9 CDRH3 sets) | `python -m paper.benchmark.check_picks` | Reproduced |
| A.4 stricter readout (36, 33, 35; 10, 4, 8; 5, 3, 12) | `python -m paper.benchmark.strict summary` | Reproduced |
| A.4 CDRH3 homopolymers (17.2 and 9.7 points) | `python -m paper.benchmark.homopolymer summary` | Reproduced |
| A.4 natural scale (19.5 times) | `python -m paper.benchmark.natural_scale compare` | Reproduced |
| A.4 windowed steering (11 and 9; 14 and 11; 41 and 25) | `python -m paper.tables.benchmark_tests` | Reproduced |
| A.4 generation quality by set | `python -m paper.analyses.quality.by_set pool` | Reproduced |
| A.8 joint steering | `python -m paper.tables.joint_steering` | Reproduced |
| A.11 binder status | `python -m paper.analyses.binder.probe report --tex manuscript.tex`; rerun: `paper/scripts/binder/probe.sh` | Reproduced (10/10 numbers) |
| Other numbers | the stage commands in `README.md` | Code present; deposited outputs |
