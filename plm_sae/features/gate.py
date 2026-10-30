"""Feature identification (Methods 3.2): which SAE latents track a sequence label such as the heavy J gene.

1. Each sequence's latents are averaged over its positions (files written by plm_sae.activations.encode).
2. A multinomial logistic-regression probe is fitted on the training sequences (MinMax scaling, balanced class
   weights, 3-fold CV over C, refit on macro-F1).
3. For each class, the top_k latents by |probe weight| are scored on the validation sequences as one-latent classifiers
   over a threshold grid (best F1 per latent).
4. Margin sweep (plm_sae.features.margin_sweep, Eqs 6-7): a latent is a feature of a class if its F1 exceeds the F1 of
   predicting every sequence as the class (F1_all+) by the margin.

Outputs in OUT_DIR: best_model_pipeline.pkl, weights/weights_class_<C>.csv, f1_scores/f1_scores_class_<C>.csv,
margin_sweep/.

Run: python -m plm_sae.features.gate TRAIN_DIR TRAIN_CSV VAL_DIR VAL_CSV LABEL_COL OUT_DIR [--top_k N]
     [--sample_n N --sample_seed S] [--thresholds ...] [--margins ...] [--cpus N]
(scripts/identification/run_gate.sh sets the paper's settings: every latent scored, 10,000 training sequences, and
the ANARCI heavy-chain J call, j_gene_anarci_heavy, as the J-gene label.)"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from plm_sae.features.margin_sweep import run_margin_sweep


def _orig_row(f):
    return int(os.path.basename(f).split(".")[0].split("_")[0])


def load_latents(latent_dir, sample_n=None, seed=0):
    """(matrix [sequences, latents], orig_rows) from <orig_row>.npy files; per-token files are averaged over tokens.
    sample_n draws that many files at random (numpy default_rng(seed)), kept in orig_row order."""
    files = sorted(glob.glob(os.path.join(latent_dir, "*.npy")), key=_orig_row)
    if sample_n and 0 < sample_n < len(files):
        idx = np.random.default_rng(seed).choice(len(files), size=sample_n, replace=False)
        files = sorted([files[i] for i in idx], key=_orig_row)

    def one(f):
        a = np.load(f, mmap_mode="r")
        return np.asarray(a.mean(axis=0) if a.ndim == 2 else a, dtype=np.float32)

    return np.vstack([one(f) for f in files]), [_orig_row(f) for f in files]


def load_labels(label_csv, orig_rows, label_col):
    """Labels of the given orig_rows, allele suffix removed (IGHJ4*02 -> IGHJ4)."""
    df = pd.read_csv(label_csv)
    df["orig_row"] = pd.to_numeric(df["orig_row"], errors="raise").astype(int)
    if df["orig_row"].duplicated().any():
        raise ValueError(f"duplicate orig_row values in {label_csv}")
    df = df.set_index("orig_row", drop=False)
    missing = sorted(set(map(int, orig_rows)) - set(df.index.values))
    if missing:
        raise ValueError(f"{len(missing)} orig_rows missing from {label_csv}, e.g. {missing[:10]}")
    return df.loc[list(map(int, orig_rows)), label_col].astype(str).str.split("*", n=1, expand=False).str[0].values


def train_probe(X, y, cpus):
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("lr", LogisticRegression(random_state=42, max_iter=1000, n_jobs=1, class_weight="balanced",
                                  multi_class="multinomial", solver="saga", tol=1e-3)),
    ])
    search = GridSearchCV(pipe, param_grid={"lr__C": np.logspace(-3, 3, 3)},
                          cv=StratifiedKFold(3, shuffle=True, random_state=42),
                          scoring={"macro_f1": "f1_macro", "acc": "accuracy",
                                   "bal_acc": make_scorer(balanced_accuracy_score)},
                          refit="macro_f1", n_jobs=cpus, verbose=1)
    search.fit(X, y)
    print("[probe] best", search.best_params_, f"cv macro-F1 {search.best_score_:.4f}", flush=True)
    return search.best_estimator_


def _class_weights(lr, i):
    """Per-class weight vector; sklearn keeps one row for a binary problem (classes_[1] against classes_[0])."""
    if lr.coef_.shape[0] == 1 and len(lr.classes_) == 2:
        return lr.coef_[0] if i == 1 else -lr.coef_[0]
    return lr.coef_[i]


def write_weights(pipe, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lr = pipe.named_steps["lr"]
    for i, cls in enumerate(lr.classes_):
        w = _class_weights(lr, i)
        idx = np.argsort(w)[::-1]
        pd.DataFrame({"feature_idx": idx, "weight": w[idx]}).to_csv(
            os.path.join(out_dir, f"weights_class_{cls}.csv"), index=False)


def latent_f1(X_raw, y, pipe, thresholds, top_k, out_dir):
    """Best single-latent F1 (latent value after the probe's MinMax scaling >= threshold) of the top_k latents by
    |weight| of each class."""
    os.makedirs(out_dir, exist_ok=True)
    X, lr = pipe.named_steps["scaler"].transform(X_raw), pipe.named_steps["lr"]
    for i, cls in enumerate(lr.classes_):
        w = _class_weights(lr, i)
        y_true, recs = (y == cls), []
        for feat in np.argsort(np.abs(w))[::-1][:top_k]:
            vals = np.clip(X[:, feat].astype(np.float32), 0.0, 1.0)
            best_f1, best_thr = 0.0, None
            for thr in thresholds:
                f1 = f1_score(y_true, vals >= thr)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            recs.append({"feature_idx": int(feat), "weight": float(w[feat]), "best_threshold": best_thr,
                         "best_f1": float(best_f1)})
        pd.DataFrame(recs).sort_values(by="best_f1", ascending=False).to_csv(
            os.path.join(out_dir, f"f1_scores_class_{cls}.csv"), index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("train_dir", "train_csv", "val_dir", "val_csv", "label_col", "out_dir"):
        p.add_argument(a)
    p.add_argument("--top_k", type=int, default=500, help="latents scored per class (the paper: all)")
    p.add_argument("--sample_n", type=int, default=None, help="training sequences (the paper: 10,000 or 6,000)")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.5, 0.8, 0.9])
    p.add_argument("--margins", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25])
    p.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    X_train, train_rows = load_latents(a.train_dir, a.sample_n, a.sample_seed)
    X_val, val_rows = load_latents(a.val_dir)
    y_train = load_labels(a.train_csv, train_rows, a.label_col)
    y_val = load_labels(a.val_csv, val_rows, a.label_col)

    pkl = os.path.join(a.out_dir, "best_model_pipeline.pkl")
    if os.path.exists(pkl):
        pipe = pickle.load(open(pkl, "rb"))   # resume after an interrupted run
    else:
        pipe = train_probe(X_train, y_train, a.cpus)
        pickle.dump(pipe, open(pkl, "wb"))
    print(classification_report(y_val, pipe.predict(X_val), zero_division=0))
    write_weights(pipe, os.path.join(a.out_dir, "weights"))
    latent_f1(X_val, y_val, pipe, a.thresholds, a.top_k, os.path.join(a.out_dir, "f1_scores"))
    run_margin_sweep(f1_dir=os.path.join(a.out_dir, "f1_scores"), y=y_val, margins=a.margins,
                     out_dir=os.path.join(a.out_dir, "margin_sweep"))


if __name__ == "__main__":
    main()
