"""SAE latents of labelled sequences, the input of feature identification.

Each sequence is run through the SAE's base model; the SAE encodes the activations at its own site:
  hyperparameter-matched SAEs   final-layer-norm output (hidden_states[-1]) divided by the model's norm factor
  sparsify TopK SAEs            raw output of the transformer layer the SAE was trained on (Section 4.1)

--pool tokens         one file per sequence, <orig_row>.npy, [tokens, dictionary size] (IMGT localisation, p-IgGen gates)
--pool mean           one file per sequence, <orig_row>.npy, mean over all padded positions (IgLM, ProGen2-OAS gates)
--pool nonzero_mean   csr shards <split>_shardNNN.npz + _rows.npy (row positions in the CSV): per latent, the mean of
                      its non-zero (> 1e-6) activations over the residue positions only (sparsify SAEs; Section 4.1)

Batch TopK and Ordered SAEs select k * N latents over everything passed to one encode() call, so they are encoded one
sequence at a time (--batch_size 1, enforced).

Run: python -m plm_sae.activations.encode SAE CSV OUT_DIR [--pool tokens|mean|nonzero_mean] [--seq_col VHorVHH]
     [--batch_size 1] [--sample N --seed S] [--shard I --num_shards N] [--split train|val]
(scripts/extraction/encode.sh sets the paper's settings.)"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from plm_sae import models, sae as S

BATCH_POOLED = ("BatchTopKSAE", "NestedDropoutBatchTopKSAE")


def read_rows(csv, seq_col, sample=None, seed=0):
    """(orig_row, sequence) pairs; --sample draws N rows with the Hugging Face datasets shuffle (as the paper's
    light-chain-type subsets were drawn)."""
    if sample:
        from datasets import load_dataset
        ds = load_dataset("csv", data_files=csv, split="train").shuffle(seed=seed).select(range(sample))
        return [int(r) for r in ds["orig_row"]], [str(s) for s in ds[seq_col]]
    df = pd.read_csv(csv, usecols=["orig_row", seq_col])
    return df["orig_row"].astype(int).tolist(), df[seq_col].astype(str).tolist()


class Site:
    """Activations at the SAE's input site for a batch of token tensors."""

    def __init__(self, lm, meta):
        self.lm, self.sparsify = lm, "sparsify_cfg" in meta
        self.norm = None if self.sparsify else float(meta["norm_factor"])
        if self.sparsify:
            self.cap = []
            layer = lm.gpt_neox.layers[int(meta["layer"])]
            layer.register_forward_hook(lambda m, i, o: self.cap.append((o[0] if isinstance(o, tuple) else o).detach()))

    def __call__(self, toks):
        with torch.no_grad():
            if self.sparsify:
                self.cap.clear()
                self.lm(input_ids=toks["input_ids"])
                return self.cap[0].float()
            out = self.lm(**toks, output_hidden_states=True)
            return out.hidden_states[-1].to(torch.float32) / self.norm


def encode_files(sae, site, model, tok, rows, seqs, out_dir, pool, batch_size, device):
    for s in range(0, len(seqs), batch_size):
        toks = {k: v.to(device) for k, v in models.tokenize(model, tok, seqs[s:s + batch_size]).items()}
        x = site(toks)
        B, T, D = x.shape
        with torch.no_grad():
            z = sae.encode(x.reshape(B * T, D)).reshape(B, T, -1)
        if pool == "mean":
            z = z.mean(dim=1)
        z = z.cpu().numpy().astype(np.float32)
        for j, r in enumerate(rows[s:s + batch_size]):
            np.save(os.path.join(out_dir, f"{r:06d}.npy"), z[j])
        if (s // batch_size) % 2000 == 0:
            print(f"[{s + B}/{len(seqs)}]", flush=True)


def encode_nonzero_mean(sae, site, tok, seqs, out_dir, split, shard, num_shards, batch_size, device):
    """Section 4.1 pooling: residue positions 1..len(sequence) only; mean of the non-zero top-k activations."""
    rows = np.arange(shard, len(seqs), num_shards)
    D, mats = sae.W_enc.shape[0], []
    for s in range(0, len(rows), batch_size):
        r = rows[s:s + batch_size]
        toks = {k: v.to(device) for k, v in models.tokenize("piggen", tok, [seqs[i] for i in r]).items()}
        x = site(toks)
        B, T, _ = x.shape
        L = torch.tensor([len(seqs[i]) for i in r], device=device)
        with torch.no_grad():
            v, i = sae.topk(x)
            pos = torch.arange(T, device=device)[None, :]
            v = v * ((pos >= 1) & (pos <= L[:, None]))[..., None]
            nz = (v > 1e-6).float()
            total = torch.zeros(B, D, device=device).scatter_add_(1, i.reshape(B, -1), (v * nz).reshape(B, -1))
            count = torch.zeros(B, D, device=device).scatter_add_(1, i.reshape(B, -1), nz.reshape(B, -1))
            mats.append(sp.csr_matrix((total / count.clamp(min=1)).cpu().numpy().astype(np.float32)))
    sp.save_npz(os.path.join(out_dir, f"{split}_shard{shard:03d}.npz"), sp.vstack(mats).tocsr())
    np.save(os.path.join(out_dir, f"{split}_shard{shard:03d}_rows.npy"), rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sae"); p.add_argument("csv"); p.add_argument("out_dir")
    p.add_argument("--pool", choices=["tokens", "mean", "nonzero_mean"], default="tokens")
    p.add_argument("--seq_col", default="VHorVHH")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="val", help="file prefix of --pool nonzero_mean shards")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    a = p.parse_args()

    meta = S.info(a.sae)
    model = {"p-IgGen": "piggen", "IgLM": "IgLM", "ProGen2-OAS": "progen2_oas"}[meta["model"].split(" ")[0]]   # json wording
    if meta.get("dict_class", "").rsplit(".", 1)[-1] in BATCH_POOLED and a.batch_size != 1:
        raise SystemExit("Batch TopK and Ordered SAEs are encoded one sequence at a time (--batch_size 1)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm, tok = models.load(model, device)
    sae = S.load(a.sae).to(device)
    site = Site(lm, meta)
    os.makedirs(a.out_dir, exist_ok=True)
    rows, seqs = read_rows(a.csv, a.seq_col, a.sample, a.seed)
    print(f"encode {a.sae} ({model}) {len(seqs)} sequences, pool {a.pool} -> {a.out_dir}", flush=True)
    if a.pool == "nonzero_mean":
        encode_nonzero_mean(sae, site, tok, seqs, a.out_dir, a.split, a.shard, a.num_shards, 16, device)
    else:
        done = {f[:-4] for f in os.listdir(a.out_dir) if f.endswith(".npy")}
        todo = [(r, s) for r, s in zip(rows, seqs) if f"{r:06d}" not in done]
        if todo:
            r, s = map(list, zip(*todo))
            encode_files(sae, site, model, tok, r, s, a.out_dir, a.pool, a.batch_size, device)
    print("done", flush=True)


if __name__ == "__main__":
    main()
