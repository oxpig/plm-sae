"""plm_sae: sparse autoencoders (TopK, Batch TopK, Ordered) on antibody language models (p-IgGen, IgLM, ProGen2-OAS).
The pipeline: SAE training, activation extraction, feature identification and steering, including output-selected
latents. Code that reproduces the paper (BIOINF-2026-1349) is in the separate `paper` package; see README.md."""
import os as _os
import sys as _sys

__version__ = "1.0.0"

# The SAE classes come from the dictionary_learning fork in third_party/, imported as the top-level package
# `dictionary_learning` (the name the checkpoints' json files give for each class).
_THIRD_PARTY = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "third_party")
if _os.path.isdir(_THIRD_PARTY) and _THIRD_PARTY not in _sys.path:
    _sys.path.insert(0, _THIRD_PARTY)
