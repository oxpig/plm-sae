"""Latent ablation during p-IgGen generation (Supplementary Section A.9, Supplementary Tables 17 and 18): one latent
(the highest-F1 latent of a gene) or a co-activation group (several latents, ablated together).

At every generation step the final-layer-norm output h is encoded with the SAE and the latents' contribution is
removed: h' = h + decode(acts with the latents scaled by 1 - strength) - decode(acts) = h - strength * sum_f acts_f
W_dec[f]. Strength 0 leaves h unchanged (the control); strength 1 removes the latents. The SAE's biases are scaled by
the norm factor so that it takes the raw final-layer-norm output. Seed 0, 1,000 sequences, p-IgGen's own sampling
(temperature 1.25, top-p 0.95); the libraries are then numbered with ANARCI.

Run: python -m paper.analyses.ablation.ablate --sae SAE --latents L [L ...] --gene IGHJk --output_dir DIR
     [--strengths 0 1] [--total 1000] [--batch_size 900] [--seed 0]      (paper/scripts/ablation/ablate.sh)"""
import argparse
import os
import random
import time

import numpy as np
import torch
from piggen.model import pIgGen

from plm_sae import models, sae as S
from plm_sae.numbering.number import number_library, paths


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Ablation:
    """Forward hook on the final layer norm that removes the latents' decoded contribution from its output."""

    def __init__(self, model, sae, latents, strength):
        self.sae, self.latents, self.strength = sae, latents, strength
        self.hook = model.gpt_neox.final_layer_norm.register_forward_hook(self._hook)

    def _hook(self, module, inputs, h):
        B, L, D = h.shape
        acts = self.sae.encode(h.view(-1, D).to(torch.float32))
        baseline = self.sae.decode(acts)
        if self.strength != 0.0:
            acts = acts.clone()
            acts[:, self.latents] = acts[:, self.latents] * (1.0 - self.strength)
        return h + (self.sae.decode(acts) - baseline).view(B, L, D).to(h.dtype)

    def remove(self):
        self.hook.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sae", required=True, help="piggen_topk_r8, piggen_batchtopk_r8 or piggen_ordered_r8")
    p.add_argument("--latents", type=int, nargs="+", required=True)
    p.add_argument("--gene", required=True, help="label used in the ANARCI file names")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--strengths", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--total", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_seconds_per_strength", type=float, default=1200.0)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pig = pIgGen.__new__(pIgGen)   # the package's sampler around the pinned model revision (plm_sae.models)
    pig.model, pig.tokenizer = models.load("piggen", device)
    pig.device = device
    sae = S.load(a.sae).to(device)
    sae.sae.scale_biases(models.NORM_FACTOR["piggen"])   # raw final-layer-norm output as input
    latents = a.latents if len(a.latents) > 1 else a.latents[0]
    label = "+".join(map(str, a.latents))
    for strength in a.strengths:
        set_seed(a.seed)
        hook = Ablation(pig.model, sae, latents, strength)
        txt, csv_path, fmt = paths(a.output_dir, label, a.gene, f"{strength:.2f}", a.seed, features=a.latents,
                                   total=a.total)
        n, start = 0, time.monotonic()
        try:
            with open(txt, "w") as f:
                while n < a.total:
                    if time.monotonic() - start > a.max_seconds_per_strength:
                        print(f"WARNING: strength {strength:.2f}: only {n}/{a.total} valid sequences after "
                              f"{a.max_seconds_per_strength:.0f} s; writing what there is", flush=True)
                        break
                    seqs = pig.generate(num_return_sequences=min(a.batch_size, a.total - n), backwards=False,
                                        top_p=0.95, temp=1.25, batch_size=a.batch_size, prompt=None)
                    f.write("".join(s + "\n" for s in seqs))
                    n += len(seqs)
        finally:
            hook.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"strength {strength:.2f}: {n} sequences -> {txt}", flush=True)
        number_library(txt, csv_path, fmt)


if __name__ == "__main__":
    main()
