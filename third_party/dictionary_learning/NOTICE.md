# Third-party code: dictionary_learning

This directory holds the parts of the `dictionary_learning` package that define and train the three SAE
architectures of the paper. It keeps its original package name: the json file of each SAE checkpoint names its class
by module path (for example `dictionary_learning.trainers.nested_dropout.NestedDropoutBatchTopKSAE`). `plm_sae` puts
`third_party/` on the Python path so that `import dictionary_learning` finds this copy.

Sources

- `dictionary_learning` by Samuel Marks, Adam Karvonen and Aaron Mueller (https://github.com/saprmarks/dictionary_learning),
  MIT licence, Copyright (c) 2024 saprmarks. See `LICENSE`.
- The copy was taken from reticular-sae by John J. Yang (https://github.com/johnyang101/reticular-sae), Apache License
  2.0, Copyright 2025 John Yang (see `LICENSE-nested-dropout`; full licence text in `LICENSE-APACHE-2.0`), which adds
  the Ordered (nested-dropout) trainer `trainers/nested_dropout.py`.

Changes for this project, relative to `dictionary_learning/` of reticular-sae commit `e89f663` (each changed file
carries a notice line at its top):

- Only the files the three SAE classes and their trainers need are included: `__init__.py`, `config.py`,
  `dictionary.py`, `trainers/__init__.py`, `trainers/trainer.py`, `trainers/top_k.py`, `trainers/batch_top_k.py` and
  `trainers/nested_dropout.py`. The other modules (activation buffer, evaluation, the other trainers), the upstream
  `tests/`, `pretrained_dictionary_downloader.sh` and `requirements.txt` are left out.
- `__init__.py` and `trainers/__init__.py`: import only the included modules.
- `trainers/nested_dropout.py`: not in the public reticular-sae repository. It comes from John J. Yang's private
  reticular-sae code, reused with his permission (he is an author of the paper), under the same Apache License 2.0
  as reticular-sae. Changed for this project: the orthonormality term is computed only when `ortho_alpha` is not 0,
  which leaves the loss unchanged and avoids a d_sae x d_sae matrix (every SAE of the paper has `ortho_alpha = 0`).

Every other included file is identical to reticular-sae.

Classes used by the paper

| SAE | dictionary class | `sae_trainer_cfg` (configs/training/sae_trainer_cfg/) |
|---|---|---|
| TopK | `trainers.top_k.AutoEncoderTopK` | `top_k` |
| Batch TopK | `trainers.batch_top_k.BatchTopKSAE` | `batch_top_k` |
| Ordered | `trainers.nested_dropout.NestedDropoutBatchTopKSAE` (batch-pooled selection) | `ordered` |
