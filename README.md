# plm-sae

Code for *Mechanistic Interpretability of Antibody Language Models Using Sparse Autoencoders* (Haque, Turnbull,
Parsan, Parsan, Yang, Beukenhorst, Deane).

This repo trains sparse autoencoders (SAEs) on three antibody language models (p-IgGen, IgLM and ProGen2-OAS), finds
SAE latents that track antibody properties such as the heavy J gene, and steers generation with them. It covers
three SAE architectures: TopK, Batch TopK and Ordered.

Data (datasets, SAE checkpoints, results and steered libraries) and an archived snapshot of this code are on Zenodo:
**https://doi.org/10.5281/zenodo.23813269**.

This branch (`anarci-ranking`) ranks the latents of the nine hyperparameter-matched SAEs against the ANARCI
heavy-chain J call, and its Zenodo record holds the results that follow from that ranking.

## Layout (key pieces)
- `plm_sae/`: pipeline code (activations, SAE training, feature identification, steering, ANARCI numbering).
- `scripts/`: one shell script per pipeline step.
- `examples/slurm/`: SLURM launchers for the pipeline scripts.
- `configs/`: paths and SAE training configs.
- `paper/`: everything needed to reproduce the tables and figures of the paper. See `paper/README.md`.
- `third_party/dictionary_learning/`: SAE trainers for all three architectures (fork of saprmarks/dictionary_learning).
- `docs/`: the steering protocol (`protocol.md`) and third-party licences (`third_party.md`).
- `tests/`: quick tests that need no data.

## Third-party code and licences
The SAE trainers are a fork of https://github.com/saprmarks/dictionary_learning. Ordered SAEs follow 'Enforcing
Orderedness to Improve Feature Consistency' by Wang et al. The expansion-32 TopK SAEs of Section 4.1 were trained
with https://github.com/EleutherAI/sparsify. See `docs/third_party.md` for licences. Note that IgLM is under the JHU
Academic Software License, which restricts use to non-commercial purposes and requires attribution.

## Getting started
1. Clone this repository and create the environment (repo root). `environment.yml` holds the dependencies only, at the
   versions used for the paper:
   ```sh
   git clone https://github.com/oxpig/plm-sae.git && cd plm-sae
   conda env create -f environment.yml
   conda activate plm-sae
   pip install --no-deps -e .
   ```
2. Install the base models from their own repositories into `external/`, at the versions used for the paper.
   Use `--no-deps` so they cannot change the versions pinned by `environment.yml`. Keep them out of `third_party/`,
   which is on the Python path, because a repository folder with the same name as its package would hide the package.
   - p-IgGen 1.0.0 (https://github.com/OliverT1/p-IgGen):
     ```sh
     git clone https://github.com/OliverT1/p-IgGen external/p-IgGen
     git -C external/p-IgGen checkout a4030ba7da81729c64b3198d331811ce1c91911e
     pip install --no-deps -e external/p-IgGen
     ```
   - IgLM 0.1.0 (https://github.com/Graylab/IgLM; JHU Academic Software License; the weights are in the repo):
     ```sh
     git clone https://github.com/Graylab/IgLM external/IgLM
     git -C external/IgLM checkout de819b56238b5cb9c23d7a6fec4c55d7fc0395e4
     pip install --no-deps -e external/IgLM
     ```
   - ProGen2 (https://github.com/salesforce/progen) and the `progen2-oas` checkpoint. The code is not pip-installed;
     plm-sae imports it from `progen2_dir` in `configs/paths.yaml` (default `external/progen/progen2`):
     ```sh
     git clone https://github.com/salesforce/progen external/progen
     mkdir -p external/progen/progen2/checkpoints/progen2-oas
     wget -P external/progen/progen2/checkpoints/progen2-oas \
         https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-oas.tar.gz
     tar -xvf external/progen/progen2/checkpoints/progen2-oas/progen2-oas.tar.gz \
         -C external/progen/progen2/checkpoints/progen2-oas/
     ```
   - sparsify 1.2.2 (https://github.com/EleutherAI/sparsify), only to retrain the expansion-32 TopK SAEs:
     ```sh
     git clone https://github.com/EleutherAI/sparsify external/sparsify
     git -C external/sparsify checkout v1.2.2
     pip install --no-deps -e external/sparsify
     ```
3. Install ANARCI 1.3 (https://github.com/oxpig/ANARCI). It needs the HMMER 3.3.2 from `environment.yml`.
   `setup.py install` downloads the IMGT germlines and builds the HMMs:
   ```sh
   git clone https://github.com/oxpig/ANARCI external/ANARCI
   git -C external/ANARCI checkout 79f6c575056dedef86cb8f405ebb039197923eec
   (cd external/ANARCI && python setup.py install)
   ```
4. Set where data and outputs go in `configs/paths.yaml` (or with `PLM_SAE_*` environment variables). By default
   they go in `data/` and `work/` inside the repo. `python -m plm_sae.paths` prints the paths in use.
5. p-IgGen is loaded from Hugging Face at a fixed revision. On nodes without internet access, download it into the
   Hugging Face cache (`HF_HOME`) first and set `HF_HUB_OFFLINE=1`:
   ```sh
   huggingface-cli download ollieturnbull/p-IgGen --revision 0e5c4ed4e4c6bed0a0c7e56ad6fa24c63ed56023
   ```
6. (Optional) To use the paper's datasets and SAE checkpoints, download the files of the Zenodo record into
   `<record_dir>` and run `bash paper/reproduce.sh setup <record_dir>`.
7. Check the install: `python tests/run_tests.py` (about 2 minutes, no data needed).

> `environment.yml` pins the versions used for the paper. Installing the base models **without** `--no-deps` can
> change these versions and break reproducibility.

## Pipeline
The same steps work for all three architectures. SAEs are named `<model>_<architecture>_r8`, e.g. `piggen_topk_r8`,
`iglm_ordered_r8`, `progen2oas_batchtopk_r8` (the `checkpoints/` folder of the Zenodo record). Models are `piggen`,
`IgLM` and `progen2_oas`. The header of each script explains its arguments, outputs and run time.

1. Generate model activations for SAE training:
   - `bash scripts/extraction/dump_activations.sh <model>`
2. Train the SAE (or use the checkpoints in the Zenodo record):
   - `bash scripts/training/train_sae.sh <sae_name>`
3. Generate SAE activations for the training and validation sets:
   - `bash scripts/extraction/encode.sh <sae_name> train`
   - Note: run once with `train` and once with `val`.
4. Run feature identification:
   - `bash scripts/identification/run_gate.sh <sae_name>`
   - The J-gene label is the ANARCI heavy-chain J call (`j_gene_anarci_heavy` of `vj_{train,val}_anarci.csv`, allele
     removed), the same label as in the Section 4.1 gate.
   - Top latents per J gene: `python -m plm_sae.features.rank <work_dir>/identification/gates/<sae_name> --top_n 3`
5. (Alternative to step 4) Pick output-selected latents, whose decoder vectors raise the output logits of a J gene's
   residues:
   - `python -m plm_sae.steering.output_scores <model>`
   - `python -m plm_sae.steering.pick_output <work_dir>/benchmark/probe/scores_<model>.csv`
6. Steer:
   - `bash scripts/steering/steer.sh <model> <sae_name> <latent> "<alphas>" <out_dir> [start]`
   - Example: `bash scripts/steering/steer.sh piggen piggen_batchtopk_r8 733 "0 48" out/latent733 102`
   - This writes 1,000 sequences per alpha and seed (seeds 0, 1 and 2), and next to each file an ANARCI table with
     the J gene of every sequence. Alpha 0 is the unsteered reference.
   - Note: `start` is the first steered position. The paper starts 5 positions before the latent first fires in
     natural sequences. Starting at 0 with a large alpha can give no valid sequences.
   - For ProGen2-OAS add `--tf32 --temp 1.0 --prompt 1EVQLVESGGGLVQPGGSLRLSCAAS`, as in the paper.

### Expansion-32 TopK SAEs (Section 4.1)
- Training: `bash scripts/training/train_sparsify.sh`
- SAE activations: `bash paper/scripts/identification/section41_encode.sh <task> [layer]`
- Feature identification: `bash paper/scripts/identification/section41_gate.sh [layer]`

## Running on SLURM
- `bash examples/slurm/submit.sh examples/slurm/<step>/<job>.slurm`
- Partition, GPU type, cluster and log directory come from `PLM_SAE_SLURM_*` environment variables.
- Jobs skip outputs that already exist, so failed jobs can be resubmitted as they are.
- See `examples/slurm/README.md`.

## Tests
- `python tests/run_tests.py` (about 2 minutes, no data needed)

## Licence and citation
The code is under the BSD 3-Clause licence (`LICENSE`). Parts that come from other projects keep their own licences,
listed in `THIRD_PARTY_NOTICES.md`. Please cite the paper (`CITATION.cff`).
