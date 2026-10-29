# Third-party notices

The code of this repository is under the BSD 3-Clause licence (`LICENSE`). Parts of it derive from other projects
and keep their own licences:
- third_party/dictionary_learning/ (vendored, trimmed to the modules the paper's SAEs use): MIT
  (saprmarks/dictionary_learning, third_party/dictionary_learning/LICENSE), with the nested-dropout trainer under
  Apache-2.0 (reticular-sae; third_party/dictionary_learning/LICENSE-nested-dropout and
  third_party/dictionary_learning/LICENSE-APACHE-2.0). Changes: third_party/dictionary_learning/NOTICE.md.
- plm_sae/training/ train.py, trainer.py, datamodule.py, utils.py and configs/training/: Apache-2.0 (reticular-sae,
  modified; plm_sae/training/NOTICE.md).
- plm_sae/steering/decode.py, the function _top_p and the order of the logit processing in sample(): Apache-2.0,
  adapted from Hugging Face transformers 4.57 (TopPLogitsWarper and GenerationMixin._sample, Copyright 2020 The
  HuggingFace Inc. team; licence text in third_party/dictionary_learning/LICENSE-APACHE-2.0).
Third-party models and tools used by this code (p-IgGen, IgLM, ProGen2, ANARCI, FlashTAP, AbLang-2, ESM-2) are not
distributed here and remain under their own licences (docs/third_party.md).
