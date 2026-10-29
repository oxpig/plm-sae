"""SAE checkpoints: <checkpoints dir>/<name>.safetensors (the weights) and <name>.json (what the SAE is).

The data record holds every SAE of the paper in this form; plm_sae.training writes it at the end of a run. The json
gives the architecture (`dict_class`, a dictionary_learning class) or marks a sparsify TopK SAE (`sparsify_cfg`) or a
steering-direction file (`kind`), plus model, layer, site and dictionary size.

  info(name)        the json, as a dict
  load(name)        the SAE as a torch module with .encode(x) -> latents, on CPU in eval mode
  decoder(name)     decoder vectors, [dictionary size, d_model] float32 tensor, one row per latent (not normalised)
  names()           every SAE in the checkpoints directory

`name` is a checkpoint name of the record (e.g. piggen_batchtopk_r8) or a path to a .safetensors file.

Encoding follows the paper: dictionary_learning SAEs encode with use_threshold=False (batch-pooled classes select
k * N latents over everything passed to one encode() call, so callers pass one sequence at a time); sparsify TopK SAEs
compute topk_k(relu((x - b_dec) @ W_enc.T + b_enc))."""
import importlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from plm_sae import paths as P


def _files(name):
    p = Path(name)
    if p.suffix == ".safetensors":
        return p, p.with_suffix(".json")
    d = P.checkpoints()
    return d / f"{name}.safetensors", d / f"{name}.json"


def info(name):
    return json.loads(_files(name)[1].read_text())


def names():
    return sorted(p.stem for p in P.checkpoints().glob("*.safetensors"))


def _class(dotted):
    module, cls = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


class SparsifyTopK(torch.nn.Module):
    """TopK SAE in the EleutherAI sparsify format (the expansion-32 SAEs of Section 4.1)."""

    def __init__(self, sd, k):
        super().__init__()
        self.W_enc = torch.nn.Parameter(sd["encoder.weight"])
        self.b_enc = torch.nn.Parameter(sd["encoder.bias"])
        self.b_dec = torch.nn.Parameter(sd["b_dec"])
        self.W_dec = torch.nn.Parameter(sd["W_dec"])
        self.k = k

    def topk(self, x):
        """(values, indices) of the k largest of relu((x - b_dec) @ W_enc.T + b_enc)."""
        return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc).topk(self.k, dim=-1)

    def encode(self, x):
        """Dense latents: the k largest pre-activations, zero elsewhere."""
        v, i = self.topk(x)
        return torch.zeros(*x.shape[:-1], self.W_enc.shape[0], dtype=v.dtype, device=v.device).scatter_(-1, i, v)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


class _DictionaryLearning(torch.nn.Module):
    def __init__(self, sae):
        super().__init__()
        self.sae = sae

    def encode(self, x):
        return self.sae.encode(x, use_threshold=False)

    def decode(self, f):
        return self.sae.decode(f)


def load(name):
    st, _ = _files(name)
    meta, sd = info(name), load_file(str(st))
    if "sparsify_cfg" in meta:
        return SparsifyTopK(sd, int(meta["k"])).eval()
    if "dict_class" not in meta:
        raise ValueError(f"{name} is not an SAE ({meta.get('kind', 'no dict_class in its json')})")
    cls = _class(meta["dict_class"])
    sae = cls(activation_dim=int(meta["activation_dim"]), dict_size=int(meta["dictionary_size"]), k=int(meta["k"]))
    missing, unexpected = sae.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return _DictionaryLearning(sae.eval()).eval()


def decoder(name):
    """Decoder vectors [dictionary size, d_model] in float32: W_dec, or decoder.weight transposed (AutoEncoderTopK)."""
    sd = load_file(str(_files(name)[0]))
    w = sd["W_dec"] if "W_dec" in sd else sd["decoder.weight"].T
    return w.float().contiguous()
