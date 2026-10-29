# Third-party models and tools

None of the base models, their code or weights, nor ANARCI, FlashTAP or AbLang-2 are distributed with this repository.
Install them from their own sources under their own licences. The versions and checksums below are the ones that
produced the paper's results.

| component | version used | used for | install |
|---|---|---|---|
| p-IgGen | package `p-iggen` 1.0.0; weights `ollieturnbull/p-IgGen`, Hugging Face snapshot `0e5c4ed4e4c6bed0a0c7e56ad6fa24c63ed56023` | base model (activations, steering) | clone `OliverT1/p-IgGen` at commit `a4030ba` and `pip install --no-deps -e` it (README.md; not on PyPI); the code loads the snapshot by revision (section 1) |
| IgLM | package `iglm` 0.1.0 (weights shipped inside the package) | base model; likelihoods in Table 3 | clone `Graylab/IgLM` at commit `de819b5` (identical to the PyPI 0.1.0 package, weights included) and `pip install --no-deps -e` it (README.md; JHU Academic Software License, non-commercial academic use) |
| ProGen2-OAS | salesforce/progen code; checkpoint `progen2-oas` | base model | section 2 |
| ANARCI | 1.3, with HMMER 3.3.2 | IMGT numbering, germline J assignment, every readout | section 3 |
| FlashTAP | `flash_abb` 0.0.4 (with its own ANARCI 2026.2.13.2 and HMMER 3.4) | developability metrics (PSH, PPC, PNC, SFvCSP) | separate environment, `envs/flashtap.yml` |
| AbLang-2 | `ablang2` 0.2.1 | naturalness in Table 3 and the steered-generation tables | separate environment, `envs/ablang2.yml` |
| ESM-2 | `facebook/esm2_t12_35M_UR50D`, snapshot `6fbf070e65b0b7291e7bbcd451118c216cff79d8` | mean per-residue log-likelihood in the generation-quality metrics | Hugging Face download, loaded by revision |
| EleutherAI sparsify | `eai-sparsify` 1.2.2 | training the expansion-32 TopK SAEs of Section 4.1 (`plm_sae.training.sparsify_topk`) | clone `EleutherAI/sparsify` at tag `v1.2.2` (identical to the PyPI release) and `pip install --no-deps -e` it (README.md) |

On nodes without internet access, download the Hugging Face snapshots into the cache (`HF_HOME`) first and set
`HF_HUB_OFFLINE=1` in the job.

## 1. p-IgGen

`plm_sae.models` loads the Hugging Face model `ollieturnbull/p-IgGen` at revision
`0e5c4ed4e4c6bed0a0c7e56ad6fa24c63ed56023` with `transformers`. The `piggen` package (1.0.0, GitHub commit `a4030ba`)
provides the sampler and output check used for generation. To download the snapshot in advance:

    huggingface-cli download ollieturnbull/p-IgGen --revision 0e5c4ed4e4c6bed0a0c7e56ad6fa24c63ed56023

## 2. ProGen2-OAS

Clone the ProGen2 code and put the `progen2-oas` checkpoint (from the download link in the ProGen2 README) in its
`progen2/checkpoints/progen2-oas/` folder. `progen2_dir` in `configs/paths.yaml` (default `external/progen/progen2`)
points at the `progen2/` folder:

    git clone https://github.com/salesforce/progen external/progen
    # download progen2-oas.tar.gz as described in external/progen/progen2/README.md and unpack it into
    # external/progen/progen2/checkpoints/progen2-oas/  (config.json, pytorch_model.bin)

The generator (`plm_sae.steering.decode`) runs its own sampling loop over forward passes, so it works with the
unmodified upstream `modeling_progen.py` (salesforce/progen commit `e8adc55`, md5 87d2e224224d764f227cd7f82015b21f). The
paper's copy of that file additionally inherited `transformers` `GenerationMixin`, which only Hugging Face `generate()`
needs; no result used it.

Checksums of the copy used for the paper:

| file | md5 |
|---|---|
| `progen2/models/progen/modeling_progen.py` | f07463f7b1a29e7700a46ff26c8d0edf |
| `progen2/models/progen/configuration_progen.py` | 9ad3b01d7c170f95c2bcfd93344d96c8 |
| `progen2/tokenizer.json` | dbbd38c5ea09e5ca273b2c3d3c73322e |
| `progen2/checkpoints/progen2-oas/pytorch_model.bin` (1,530,726,083 bytes) | 53240904b73793264e18f4c143a137fd |
| `progen2/checkpoints/progen2-oas/config.json` | 9422ff051b29d80d1a745aa7dd79a8a5 |


## 3. ANARCI

ANARCI 1.3 from the OPIG repository (https://github.com/oxpig/ANARCI), installed into the main environment after
`environment.yml` (which pins HMMER 3.3.2):

    git clone https://github.com/oxpig/ANARCI external/ANARCI && cd external/ANARCI
    git checkout 79f6c575056dedef86cb8f405ebb039197923eec   # latest commit (May 2024), setup.py version 1.3
    python setup.py install   # downloads the IMGT germlines and builds the HMMs

`plm_sae.numbering` calls ANARCI in-process (IMGT scheme, `assign_germline=True`) and
runs HMMER single-threaded per worker (`HMMER_NCPU=1`). No species restriction is passed; the human-chain readout
filters on the assigned species afterwards.

## 4. IgLM

`plm_sae` calls IgLM through the installed `iglm` package (`IgLM(model_name="IgLM")`). The md5 of the weights shipped
in `iglm/trained_models/IgLM/pytorch_model.bin` was `37f3fd85283ba037f9a256f1eabeeb42`. IgLM is distributed under the
JHU Academic Software License; see the IgLM repository for its terms.

## 5. FlashTAP and AbLang-2

These run as separate processes in their own environments; `configs/paths.yaml` keys `python_flashtap` and
`python_ablang` name the interpreters (default: the interpreter running `plm_sae`). FlashTAP scores only pairs whose
chains contain the 20 standard amino acids (`paper.analyses.properties.outsel.tap`).
