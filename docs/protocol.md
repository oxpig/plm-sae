# Steering protocol

This page states the steering benchmark and the output selection exactly as the code applies them. Manuscript
references: Methods "Steering", "Steering benchmark" and "Output-selected latents"; Supplementary "Hyperparameter-matched
SAEs and steering benchmark", "Steering site and position gate", "Output-selected latents", "Graded steering tasks" and
"Developability liabilities". The numbers below are constants in the code; nothing here is a tunable default.

## 1. Steering

A latent `i` is represented by its unit decoder vector `d_i` (row `i` of `W_dec`, normalised). Steering adds `alpha * d_i`
to the hidden state that the SAE encodes, at every generated position from the start position on:

    h* = h + alpha * d_i

* **Site.** The final layer-norm output that the matched SAEs read (`--site post`): p-IgGen and IgLM layer 3 (the last
  block), ProGen2-OAS layer 26. `alpha` is in units of that tensor; the SAEs see it divided by the model's normalisation
  factor (`plm_sae.models.NORM_FACTOR`: p-IgGen 27.1673, IgLM 24.0302, ProGen2-OAS 71.1121).
* **Generator.** `plm_sae.steering.generate` generates many (latent, alpha) configurations in one batch, one steering
  vector per row. Rows are laid out j-major (row `j` of every configuration before row `j+1` of any) and draw common
  random numbers, so configurations that share a seed share their sampling noise. fp32 throughout (ProGen2-OAS: TF32
  matmuls). Libraries are rejection-sampled until 1,000 sequences pass the model's output check; a configuration that
  adds fewer new valid sequences than 8% of the rows spent on it in a round is abandoned (`--min_yield 0.08`). Seeds 0, 1
  and 2 run one after another, each from `torch.manual_seed(seed)`.
* **Sampling.**

  | model | prompt | temperature | top-p | max length |
  |---|---|---|---|---|
  | p-IgGen | `1` (start token) | 1.25 | 0.95 | 400 new tokens |
  | IgLM | `[HEAVY][HUMAN]EVQ` | 1.0 | 1.0 | 150 tokens |
  | ProGen2-OAS | `1EVQLVESGGGLVQPGGSLRLSCAAS` (human FR1) | 1.0 | 0.95 | 256 tokens |

  For ProGen2-OAS the generator's defaults are the model's own (temperature 0.2, prompt `1EVQ`); the reported runs pass
  `--tf32 --temp 1.0 --prompt 1EVQLVESGGGLVQPGGSLRLSCAAS`.
* **Library files.** `runs/<model>/<SAE>/<set>/<rule>/<gene>/steered_l3_f<latent>_a<alpha>_seed<seed>_seq1000.txt`, one
  sequence per line (`l3` is the layer label used for every model). `rule` is the start rule (`gated` or `ungated`,
  section 4); the ProGen2-OAS run at temperature 1.0 writes `runs/progen2_oas/<SAE>/<set>/<gene>/`.

Libraries are byte-identical across reruns on the same GPU type with the same configuration file: the configuration
set, its order and the batch size (`--rows`, by default fixed by GPU memory) decide which random numbers each row draws.
A rerun with other task files draws other random numbers, so it reproduces the protocol, not the deposited bytes.

## 2. Readout

Every unique sequence is numbered once with ANARCI (IMGT scheme, germline assignment; `plm_sae.numbering`). A steered
library is read as the human heavy chains whose J region is at least 50% identical to the assigned germline J gene
("J identity >= 0.5"); chains without a recognisable J segment, which ANARCI assigns to IGHJ2, are removed by this
filter. The target proportion is the share of these chains assigned to the target gene. The stricter readout of
Supplementary "Output-selected latents" (identity >= 0.8, proportion over all generated sequences) is
`paper.benchmark.strict`.

## 3. Latent sets (per SAE, per heavy J gene IGHJ1 to IGHJ6, three latents each)

| set | label in result files | selection |
|---|---|---|
| highest F1 | `f1` | the three latents of highest validation F1 for the gene (matched SAEs: F1 of the mean activation over all positions against the dataset's J labels; expansion-32 TopK SAE: against ANARCI-assigned J genes) |
| baseline | `baseline` | random latents that fire in 5 to 50% of the selection pool, at a rate within 1.25-fold of their overall rate for every J gene, and not steered in any other set |
| output-selected | `output` | the three highest output scores (section 5) among latents firing in at least 1% of the same selection pool |

Sets steered without the position gate carry the suffix ` ungated` in `benchmark/final_latents.csv`. A latent fires
in a sequence if its activation is above zero at any position. Sets are drawn per SAE, so the 18 latents of a set cover
the six genes. Baseline draws 1 and 2-3 are `positions/<model>_controls.csv` and `_controls2.csv`
(`paper.benchmark.baselines`).

**Selection pool.** Baseline eligibility and the output-selection firing filter use one draw of validation heavy
chains: 300 per IGHJ1-6 gene (1,800 in all) from `vj_val_anarci.csv`, each gene's sample drawn with
`random_state=1` (`groupby("gene").apply(lambda d: d.sample(n=300, random_state=1))`; note that pandas'
`groupby("gene").sample(n=300, random_state=1)` draws a different set). One function draws it,
`plm_sae.steering.selection_pool.validation_pool` (constants `SEED` and `PER_GENE`); `paper.benchmark.baselines`,
`plm_sae.steering.output_scores` and `paper.analyses.properties.outsel.scores` call it. On this pool the output picks
are the top-3 sets of all 54 J-gene targets, 9 CDRH3 targets and 12 developability targets. One steered baseline latent
sits exactly at the eligibility boundary (p-IgGen TopK 3080, IGHJ2: enrichment 1.25 in floating point).
`python -m paper.benchmark.check_picks` reproduces the picks from the deposited scores without a GPU.

## 4. Start position (position gate)

* **Gated (p-IgGen, ProGen2-OAS):** the highest-F1 latent of a gene starts five positions before the 5th percentile of its
  onset over validation sequences of that gene, where the onset is the first position at which its activation reaches
  10% of that sequence's maximum (`onset_p05 - 5`, floored at 0; rule `gated`, `paper.benchmark.positions`). The three
  highest-F1 ProGen2-OAS Ordered latents that never fire on those sequences (759, 6902, 4036) start at position 0.
  For the expansion-32 TopK SAE the onset is taken over 3,000 unsteered p-IgGen sequences.
* **Baseline and output-selected latents** start at the start position of the gene's highest-F1 latent, so every set is
  compared over the same positions.
* **IgLM** is steered from position 0 for every set (ungated), as reported.

## 5. Output score

At the steering site the steered hidden state feeds only the output layer, so `d_i` adds the fixed bias `W_U d_i` to
the next-residue logits. The score of latent `i` for gene `g` is the sum, over the residues of `g`'s germline J segment,
of `W_U d_i` centred over the 20 amino acids, minus the mean of that sum over the six genes
(`plm_sae.steering.output_scores`, column `ll_pred`; selection `plm_sae.steering.pick_output`):

| gene | germline J segment |
|---|---|
| IGHJ1 | AEYFQHWGQGTLVTVSS |
| IGHJ2 | YWYFDLWGRGTLVTVSS |
| IGHJ3 | DAFDIWGQGTMVTVSS |
| IGHJ4 | YFDYWGQGTLVTVSS |
| IGHJ5 | NWFDPWGQGTLVTVSS |
| IGHJ6 | YYYYYGMDVWGQGTTVTVSS |

The score uses only the decoder vector, the unembedding matrix and the germline sequences: no generation, no
activations, no labelled sequences. The firing filter (>= 1% of the selection pool, column `seq_freq`) is the only use
of data. For CDRH3 properties and developability metrics the score is the mean bias toward the residues that raise the
property minus that toward the residues that lower it, with the same pool and baselines, and a start five positions
before the onset of the property's highest-ranked latent (`paper.analyses.properties`).

## 6. Strength ladder

| SAE | base range | step |
|---|---|---|
| p-IgGen matched SAEs | 0 to 120 | 8 |
| IgLM matched SAEs | 0 to 60 | 4 |
| ProGen2-OAS matched SAEs | 0 to 30 | 3 |
| p-IgGen TopK, expansion 32 | 0 to 240 | 16 |

The base ranges shift the amino-acid logits by a similar amount relative to each model's sampling temperature (the
expansion-32 SAE's input has 2.2 times the norm). After the base ladder:

* **Extension:** if at least 80% of sequences (2,400 of 3,000, pooled over the three seeds) still yield a qualifying
  chain at the largest alpha, the range is extended by up to two blocks of five strengths, each block spanning one further base range
  (`paper.benchmark.extend`).
* **Refinement:** if fewer than five strengths meet the 80% criterion, seven evenly spaced strengths are added below the
  first strength that does not (`paper.benchmark.make_refine`).

## 7. Criteria (per latent, three seeds of 1,000 sequences)

* `alpha*` (per seed): the largest alpha at which at least 800 of the 1,000 sequences are qualifying chains (section 2).
* **Correlation criterion:** in at least two of three seeds, Spearman's rho between the target proportion and alpha, over
  the strengths `alpha <= alpha*` (at least three strengths), is at least 0.7 with p < 0.05 (hypothesised direction:
  rising; `plm_sae.steering.criterion.judge_pair`).
* **Magnitude criterion:** the correlation criterion passes and, in at least two seeds, the target proportion at
  `alpha*` is at least 25% above its value at alpha 0 (relative rise).
* Implemented in `paper.benchmark.analyze_final.paper` (result columns `paper_corr`, `paper_mag`, `paper_astar_med`,
  `paper_base`, `paper_ach`). Tables report latents passing the correlation criterion with both criteria in parentheses.

## 8. Comparing sets

Sets are compared with an exact two-sided Mantel-Haenszel test stratified by SAE-gene pair: the conditional
distribution of the number of passing latents in one set, given the per-pair totals, is the convolution of the per-pair
hypergeometric distributions, and the p-value sums the probabilities not larger than that of the observed count
(`paper.benchmark.rsets.strat_exact`).

## 9. Labels in the result files

| label | meaning |
|---|---|
| `f1`, `baseline`, `output` | the three sets of section 3; with the suffix ` ungated`, steered from position 0 |
| `joint` | co-activation groups steered as one direction (`phi`: `phi05` or `phi07`, `members`: the group's latents) |
| `light chain`, `developability`, `section41` | light-chain-type latents, developability latents 980 and 1088, the expansion-32 TopK latents of Section 4.1 |
| `gated`, `ungated` | start rule of section 4 |
| task names | `f1_`, `bl_`, `out_` (gated) and `f1u_`, `blu_`, `outu_` (ungated) `<model>_<SAE>_<gene>`; `x<block>_` extension and `r_` refinement tasks of a parent task; `t10a_`, `t10b_`, `t10c_` the ProGen2-OAS output-selected, baseline and highest-F1 tasks at temperature 1.0 |
