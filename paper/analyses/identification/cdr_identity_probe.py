"""CDR residue-identity probe of Section 4.1: does a linear probe on the SAE latents (or on the hidden neurons) tell which
CDR, if any, a residue lies in?

Input: the paired sequence of each row ("1" + VH + VL + "2", the text column of datasets/CDR_{training,validation}.csv,
padded to 350 tokens), first 600 training and 1,000 validation rows. Site: output of p-IgGen block 3 (gpt_neox.layers[3],
before the final layer norm). Features per residue: the top-32 latents of the expansion-32 TopK SAE
(piggen_topk_r32_layer3) at that position, or the 768 hidden neurons. Label per residue, 7 classes: CDRH1-3, CDRL1-3 or
none, with IMGT boundaries (ANARCI, IMGT scheme: CDR1 27-38, CDR2 56-65, CDR3 105-117; --labels imgt, the paper) or the
dataset's own annotation (CDR_Mask; --labels dataset). One-vs-rest logistic regression (liblinear), C in {0.1, 1, 10}
by 3-fold cross-validation on accuracy, accuracy on the validation residues.
Outputs in --outdir (default <work_dir>/identification/cdr_probe): cdr_probe_<features>_<labels>_metrics.txt.
Usage: python -m paper.analyses.identification.cdr_identity_probe [--labels imgt|dataset|both] (GPU)"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from plm_sae import models, paths as P, sae as S

IMGT_CDRS = ((27, 38), (56, 65), (105, 117))


def imgt_labels(vh, vl, ncpu):
    """Per residue of vh + vl: 'A'/'B'/'C' heavy CDR1-3, 'Z'/'Y'/'W' light CDR1-3, '-' elsewhere (IMGT, ANARCI)."""
    from anarci import anarci
    out = []
    num_h, _, _ = anarci(list(enumerate(vh)), scheme="imgt", output=False, ncpu=ncpu)
    num_l, _, _ = anarci(list(enumerate(vl)), scheme="imgt", output=False, ncpu=ncpu)
    for h, l, nh, nl in zip(vh, vl, num_h, num_l):
        lab = ["-"] * (len(h) + len(l))
        for seq_num, off, syms in ((nh, 0, "ABC"), (nl, len(h), "ZYW")):
            if not seq_num:
                continue
            residues, k, _ = seq_num[0]
            for (pos, _ins), aa in residues:
                if aa == "-":
                    continue
                for s, (lo, hi) in zip(syms, IMGT_CDRS):
                    if lo <= pos <= hi:
                        lab[off + k] = s
                k += 1
        out.append("".join(lab))
    return out


def residue_table(df, bs, dev, sae, features):
    """Sparse feature matrix, one row per residue position (token positions 1..n) of the given rows."""
    model, tok = models.load("piggen", dev)
    cap = []
    model.gpt_neox.layers[3].register_forward_hook(
        lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach().float()))
    texts, mats = df["text"].tolist(), []
    for s0 in range(0, len(texts), bs):
        batch = texts[s0:s0 + bs]
        toks = tok(["1" + t + "2" for t in batch], padding="max_length", truncation=True, max_length=350,
                   return_tensors="pt")
        cap.clear()
        with torch.no_grad():
            model(input_ids=toks["input_ids"].to(dev))
            x = cap[0]                                           # (B, 350, 768)
            if features == "latents":
                v, idx = sae.topk(x)
        for b, t in enumerate(batch):
            n = min(len(t), 348)
            if features == "latents":
                vv, ii = v[b, 1:n + 1].cpu().numpy(), idx[b, 1:n + 1].cpu().numpy()
                keep = vv > 1e-6
                rows = np.repeat(np.arange(n), keep.sum(1))
                mats.append(sp.csr_matrix((vv[keep], (rows, ii[keep])), shape=(n, sae.W_enc.shape[0])))
            else:
                mats.append(sp.csr_matrix(x[b, 1:n + 1].cpu().numpy()))
        if (s0 // bs) % 20 == 0:
            print(f"  {features}: {s0 + len(batch)}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return sp.vstack(mats).tocsr()


def flat(labels):
    return np.concatenate([np.array(list(s[:min(len(s), 348)])) for s in labels])


def train_eval(Xtr, ytr, Xva, yva, tag, outdir):
    if Xtr.shape[1] > 768:                                       # latents: the ones active in training
        used = np.unique(Xtr.nonzero()[1])
        Xtr, Xva = Xtr[:, used], Xva[:, used]
    clf = LogisticRegression(multi_class="ovr", solver="liblinear", max_iter=1000)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(clf, {"C": [0.1, 1, 10]}, cv=cv, scoring="accuracy", n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    gs.fit(Xtr, ytr)
    pred = gs.best_estimator_.predict(Xva)
    acc, mf1 = accuracy_score(yva, pred), f1_score(yva, pred, average="macro")
    maj = (yva == pd.Series(ytr).mode()[0]).mean()
    print(f"{tag}: n_features={Xtr.shape[1]} best_C={gs.best_params_['C']} val_acc={acc:.4f} macro_f1={mf1:.4f} "
          f"majority_class={maj:.4f}", flush=True)
    with open(f"{outdir}/{tag}_metrics.txt", "w") as f:
        f.write(f"n_train_rows={Xtr.shape[0]}\nn_val_rows={Xva.shape[0]}\nn_features={Xtr.shape[1]}\n"
                f"best_C={gs.best_params_['C']}\nval_accuracy={acc}\nval_macro_f1={mf1}\nmajority_class_accuracy={maj}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--labels", choices=["imgt", "dataset", "both"], default="imgt")
    p.add_argument("--features", choices=["latents", "neurons", "both"], default="both")
    p.add_argument("--ncpu", type=int, default=8)
    p.add_argument("--outdir", default=str(P.work("identification", "cdr_probe")))
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = torch.device("cuda")
    sae = S.load("piggen_topk_r32_layer3").to(dev)
    tr = pd.read_csv(P.dataset("CDR_training.csv")).iloc[:a.n_train]
    va = pd.read_csv(P.dataset("CDR_validation.csv")).iloc[:a.n_val]
    for d in (tr, va):
        assert (d["text"] == d["VHorVHH"] + d["VL"]).all() and (d["text"].str.len() == d["CDR_Mask"].str.len()).all()
    lab = {}
    if a.labels in ("imgt", "both"):
        lab["imgt"] = (imgt_labels(tr.VHorVHH.tolist(), tr.VL.tolist(), a.ncpu),
                       imgt_labels(va.VHorVHH.tolist(), va.VL.tolist(), a.ncpu))
    if a.labels in ("dataset", "both"):
        lab["dataset"] = (tr.CDR_Mask.tolist(), va.CDR_Mask.tolist())
    for feats in (["latents", "neurons"] if a.features == "both" else [a.features]):
        Xtr, Xva = residue_table(tr, a.bs, dev, sae, feats), residue_table(va, a.bs, dev, sae, feats)
        for name, (ltr, lva) in lab.items():
            train_eval(Xtr, flat(ltr), Xva, flat(lva), f"cdr_probe_{feats}_{name}", a.outdir)
    print("DONE")


if __name__ == "__main__":
    main()
