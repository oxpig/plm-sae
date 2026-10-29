# Third-party code: training harness

The SAE training harness comes from reticular-sae by John J. Yang (https://github.com/johnyang101/reticular-sae),
licensed under the Apache License, Version 2.0 (Copyright 2025 John Yang; notice in
`third_party/dictionary_learning/LICENSE-nested-dropout`, full text in `third_party/dictionary_learning/LICENSE-APACHE-2.0`).
Each file below carries a notice line at its top. Status is relative to reticular-sae commit `e89f663`.

| file in this repository | file in reticular-sae | status |
|---|---|---|
| `plm_sae/training/train.py` | `training/train.py` | modified |
| `plm_sae/training/trainer.py` | `training/trainer.py` | modified |
| `plm_sae/training/datamodule.py` | `training/datamodule.py` | modified |
| `plm_sae/training/utils.py` | `training/utils.py` | modified |
| `configs/training/train.yaml`, `datamodule/base.yaml`, `sae_trainer_cfg/top_k.yaml` | `training/configs/train_default.yaml`, `datamodule/base.yaml`, `sae_trainer_cfg/top_k.yaml` | modified |
| `configs/training/datamodule/{piggen,iglm,progen2_oas}.yaml`, `sae_trainer_cfg/{batch_top_k,ordered}.yaml` | none | added for this project |

The changes for this project: the antibody language models (p-IgGen, IgLM, ProGen2-OAS) and their activation caches,
the Batch TopK and Ordered trainer classes, step limits under manual optimisation, the export of each trained SAE as
safetensors + json, package imports and configurable paths. The fidelity evaluation, Weights & Biases logging and the
trainers the paper does not use are removed.

`reconstruction.py`, `screen_fvu.py` and `sparsify_topk.py` are this project's own code. `sparsify_topk.py` uses the
EleutherAI `sparsify` library (https://github.com/EleutherAI/sparsify, MIT), which is installed separately.
