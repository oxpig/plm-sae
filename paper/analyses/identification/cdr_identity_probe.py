"""CDR residue-identity probe of the TopK SAE of Section 4.1 (Section 4.1: validation accuracy 0.96 for the latents).
Per residue of the CDR-masked sequences (CDR_Mask of datasets/CDR_{training,validation}.csv; 600 training and 1,000
validation sequences), the SAE's top-32 latents at that position (input: output of p-IgGen block 3) are the features
and the amino acid is the label: one-vs-rest logistic regression (liblinear), C in {0.1, 1, 10} by 3-fold
cross-validation on accuracy, accuracy on the validation residues.
Outputs in --outdir (default <work_dir>/identification/cdr_probe): cdr_probe_{train,val}_latents.csv.gz,
cdr_probe_metrics.txt (val_accuracy). Usage: python -m paper.analyses.identification.cdr_identity_probe (GPU)"""
import argparse, gc, os
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from plm_sae import models, paths as P, sae as S

SPECIAL = {"1", "2", "*", "X"}


def pad_sequence(seq, max_length=350):
    return f"1{seq}2".ljust(max_length, "*")


def build_latent_table(csv_path, num_seq, bs, dev, sae):
    model, tok = models.load("piggen", dev)
    cap = []
    model.gpt_neox.layers[3].register_forward_hook(
        lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach().float())
    )
    df_seq = pd.read_csv(csv_path)[["CDR_Mask"]].iloc[:num_seq]
    padded = df_seq["CDR_Mask"].apply(pad_sequence)
    aa = np.array([list(s) for s in padded])  # (num_seq, 350)
    rows = []
    for s0 in range(0, num_seq, bs):
        s1 = min(s0 + bs, num_seq)
        texts = list(padded.iloc[s0:s1])
        toks = tok(texts, padding="max_length", truncation=True, max_length=350,
                    return_tensors="pt")
        ids = toks["input_ids"].to(dev)
        cap.clear()
        with torch.no_grad():
            model(input_ids=ids)
            x = cap[0]  # raw pre-LN block output, (B, T, 768)
            v, idx = sae.topk(x)
        v = v.cpu().numpy()
        idx = idx.cpu().numpy()
        for bi in range(s1 - s0):
            seq_i = s0 + bi
            for t in range(350):
                a = aa[seq_i, t]
                if a in SPECIAL:
                    continue
                d = {f"Neuron_{int(idx[bi, t, k])}": float(v[bi, t, k]) for k in range(v.shape[-1]) if v[bi, t, k] > 1e-6}
                d["AminoAcid"] = a
                d["sequence_idx"] = seq_i
                d["position_idx"] = t
                rows.append(d)
        if (s0 // bs) % 20 == 0:
            print(f"[{csv_path}] {s0}/{num_seq}", flush=True)
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows).fillna(0.0)


def train_eval(train_df, val_df, tag, outdir):
    drop_cols = [c for c in ["AminoAcid", "sequence_idx", "position_idx"] if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["AminoAcid"]
    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    y_val = val_df["AminoAcid"]
    all_feats = X_train.columns.union(X_val.columns)
    X_train = X_train.reindex(columns=all_feats, fill_value=0)
    X_val = X_val.reindex(columns=all_feats, fill_value=0)
    clf = LogisticRegression(multi_class="ovr", solver="liblinear", max_iter=1000)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(clf, {"C": [0.1, 1, 10]}, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    y_pred = gs.best_estimator_.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    print(f"{tag}: n_features={X_train.shape[1]} best_C={gs.best_params_['C']} val_acc={acc:.4f} macro_f1={macro_f1:.4f}")
    with open(f"{outdir}/{tag}_metrics.txt", "w") as f:
        f.write(f"n_train_rows={len(train_df)}\nn_val_rows={len(val_df)}\nn_features={X_train.shape[1]}\n")
        f.write(f"best_C={gs.best_params_['C']}\nval_accuracy={acc}\nval_macro_f1={macro_f1}\n")
    return acc, macro_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--outdir", default=str(P.work("identification", "cdr_probe")))
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = torch.device("cuda")
    sae = S.load("piggen_topk_r32_layer3").to(dev)

    train_df = build_latent_table(
        P.dataset("CDR_training.csv"), a.n_train, a.bs, dev, sae
    )
    train_df.to_csv(f"{a.outdir}/cdr_probe_train_latents.csv.gz", index=False, compression="gzip")
    val_df = build_latent_table(
        P.dataset("CDR_validation.csv"), a.n_val, a.bs, dev, sae
    )
    val_df.to_csv(f"{a.outdir}/cdr_probe_val_latents.csv.gz", index=False, compression="gzip")
    gc.collect()
    train_eval(train_df, val_df, "cdr_probe", a.outdir)
    print("DONE")


if __name__ == "__main__":
    main()
