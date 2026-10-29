"""Base-model activations for SAE training: the final-layer-norm output of every token of a seeded random sample of
the training set, one LitData record ([tokens, d_model] float32) per padded sequence.

Settings of the paper's SAEs:
  piggen        oas_sae_training_dataset.csv (paired VH+VL, already wrapped as 1...2), shuffle seed 0, 100,005 records
                (1,695 chunks of 59, the records every matched p-IgGen SAE read)
  IgLM          IgLM_sae_training_dataset.csv (heavy chains, already tagged [HEAVY][HUMAN]), seed 42, 100,000 records
  progen2_oas   Progen2_oas_sae_training_dataset.csv (heavy chains, wrapped here as 1...2), seed 42, 100,000 records

Run: python -m plm_sae.activations.dump MODEL OUT_DIR [--num_seqs N] [--seed S]   (one GPU)
(scripts/extraction/dump_activations.sh MODEL sets the paper's settings.)"""
import argparse
import os

import litdata as ld
import numpy as np
import torch
from datasets import load_dataset

from plm_sae import models, paths as P

DATA = {"piggen": ("oas_sae_training_dataset.csv", 0, 100005, 128, "64MB"),
        "IgLM": ("IgLM_sae_training_dataset.csv", 42, 100000, 1, "64MB"),
        "progen2_oas": ("Progen2_oas_sae_training_dataset.csv", 42, 100000, 1, "256MB")}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model", choices=models.MODELS)
    p.add_argument("out_dir")
    p.add_argument("--num_seqs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    csv, seed, n, batch, chunk = DATA[a.model]
    seed = seed if a.seed is None else a.seed
    n = n if a.num_seqs is None else a.num_seqs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm, tok = models.load(a.model, device)
    texts = load_dataset("csv", data_files=str(P.dataset(csv)), split="train").shuffle(seed=seed).select(range(n))["text"]
    wrap = a.model == "progen2_oas"   # the p-IgGen and IgLM training sets are stored with their start/tag tokens

    def records(idx):
        toks = models.tokenize(a.model, tok, [texts[i] for i in idx], wrap=wrap, truncation=a.model != "progen2_oas")
        with torch.no_grad():
            hs = lm(**{k: v.to(device) for k, v in toks.items()}, output_hidden_states=True).hidden_states[-1]
        for row in hs.cpu().numpy().astype(np.float32):
            yield torch.from_numpy(row)

    os.makedirs(a.out_dir, exist_ok=True)
    ld.optimize(fn=records, inputs=[list(range(i, min(i + batch, n))) for i in range(0, n, batch)],
                output_dir=a.out_dir, num_workers=0, chunk_bytes=chunk, mode="overwrite")
    print(f"{n} records written to {a.out_dir}")


if __name__ == "__main__":
    main()
