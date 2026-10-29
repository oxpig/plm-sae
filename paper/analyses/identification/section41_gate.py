"""Feature identification of the expansion-32 TopK SAEs of Section 4.1 (Table 1, Supplementary Tables 12 and 13).

Input: per sequence and latent, the mean of the latent's non-zero activations over the residue positions
(paper/scripts/identification/section41_encode.sh, csr shards). Labels: ANARCI heavy J gene (j_gene_anarci_heavy),
human rows only (ANARCI species), allele removed. Steps:
  1. one-vs-rest logistic regression on the training sequences, C in {0.01, 0.1, 1} by 3-fold cross-validation on
     accuracy (StratifiedKFold with shuffle and no fixed seed, so the chosen C can vary between runs);
  2. per class, the 500 latents with the highest weight, each MinMax-scaled on the validation sequences and scored as a
     one-latent classifier over the thresholds 0.1, 0.2, 0.5, 0.8, 0.9 (best F1);
  3. margin sweep (plm_sae.features.margin_sweep, Eqs 6-7).
Outputs in OUT_DIR: full_model_evaluation_<trait>.csv (+ _summary.json), weights/class_<c>_weights.csv,
feature_f1_scores_class_<c>.csv, margin_sweep/.

Run: python -m paper.analyses.identification.section41_gate ACT_DIR OUT_DIR   (paper/scripts/identification/section41_gate.sh)"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from plm_sae import paths as P
from plm_sae.features.margin_sweep import run_margin_sweep

N_LATENTS = 24576
N_SEQS = {"train": 103055, "val": 30754}
TRAIT = "j_gene_anarci_heavy"
THRESHOLDS = [0.1, 0.2, 0.5, 0.8, 0.9]


def load_shards(act_dir, split):
    """[sequences, N_LATENTS + 1] csr matrix in CSV row order (the last column is all zero, as in the paper's runs)."""
    parts = sorted(glob.glob(f"{act_dir}/{split}_shard???.npz"))
    X = sp.vstack([sp.load_npz(f) for f in parts]).tocsr()
    rows = np.concatenate([np.load(f.replace(".npz", "_rows.npy")) for f in parts])
    n = N_SEQS[split]
    assert X.shape == (n, N_LATENTS) and len(set(rows)) == len(rows) == n, (X.shape, len(rows))
    inv = np.empty(n, dtype=np.int64)
    inv[rows] = np.arange(n)
    return sp.hstack([X[inv], sp.csr_matrix((n, 1), dtype=np.float32)]).tocsr()


def human_labels(label_csv):
    """(labels without allele, row positions) of the rows whose ANARCI heavy J species is human."""
    df = pd.read_csv(label_csv)
    idx = df.index[df["j_gene_species_anarci_heavy"] == "human"]
    y = df[TRAIT][idx].apply(lambda x: x.split("*")[0] if isinstance(x, str) and "*" in x else
                             (x.split("-")[0] if isinstance(x, str) and "-" in x else x))
    return y, idx


def train_classifier(X, y):
    search = GridSearchCV(LogisticRegression(random_state=42, multi_class="ovr", max_iter=1000, n_jobs=-1),
                          {"C": [0.01, 0.1, 1]}, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring="accuracy",
                          n_jobs=-1)
    search.fit(X.tocsc(), y)
    print(f"best C {search.best_params_['C']}, cross-validated accuracy {search.best_score_:.3f}", flush=True)
    return search.best_estimator_


def save_evaluation(clf, X, y, path):
    rep = classification_report(y, clf.predict(X), output_dict=True)
    pd.DataFrame(rep).transpose().to_csv(path, index=True)
    json.dump({"accuracy": rep["accuracy"], "macro_avg_f1": rep["macro avg"]["f1-score"],
               "weighted_avg_f1": rep["weighted avg"]["f1-score"], "best_params": clf.get_params(),
               "num_classes": len(clf.classes_), "classes": list(map(str, clf.classes_))},
              open(path.replace(".csv", "_summary.json"), "w"), indent=2)


def write_weights(clf, out_dir):
    """weights/class_<c>_weights.csv: every latent, sorted by weight (highest first)."""
    os.makedirs(out_dir, exist_ok=True)
    for i, c in enumerate(clf.classes_):
        with open(os.path.join(out_dir, f"class_{str(c).replace('/', '_').lower()}_weights.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Index", "Weight", "Feature"])
            for idx, weight in sorted(enumerate(clf.coef_[i]), key=lambda x: x[1], reverse=True):
                w.writerow([idx, weight, f"Neuron_{idx}"])


def latent_f1(X, y, target, latents, path):
    """Best one-latent F1 for `target` over THRESHOLDS, latent values MinMax-scaled on X."""
    vals = MinMaxScaler().fit_transform(np.column_stack([X[:, i].toarray().ravel() for i in latents]))
    y_true = np.where(y.reset_index(drop=True) == target, target, f"not_{target}")
    out = []
    for j, lat in enumerate(latents):
        best_f1, best_thr = 0.0, None
        for thr in THRESHOLDS:
            f1 = f1_score(y_true, np.where(vals[:, j] >= thr, target, f"not_{target}"), pos_label=target, average="binary")
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        out.append({"feature": f"Neuron_{lat}", "neuron_id": int(lat), "best_threshold": best_thr, "best_f1": best_f1})
    pd.DataFrame(out).sort_values(by="best_f1", ascending=False).to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("act_dir")
    p.add_argument("out_dir")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    y_train, i_train = human_labels(P.dataset("vj_train_anarci.csv"))
    y_val, i_val = human_labels(P.dataset("vj_val_anarci.csv"))
    X_train = load_shards(a.act_dir, "train")[i_train, :]
    X_val = load_shards(a.act_dir, "val")[i_val, :]
    print(f"train {X_train.shape}, val {X_val.shape}", flush=True)
    clf = train_classifier(X_train, y_train)
    save_evaluation(clf, X_val, y_val, os.path.join(a.out_dir, f"full_model_evaluation_{TRAIT}.csv"))
    wdir = os.path.join(a.out_dir, "weights")
    write_weights(clf, wdir)
    for c in y_val.unique():
        name = str(c).replace("/", "_").lower()
        top = pd.read_csv(os.path.join(wdir, f"class_{name}_weights.csv"))["Index"].head(500).tolist()
        latent_f1(X_val, y_val, c, top, os.path.join(a.out_dir, f"feature_f1_scores_class_{name}.csv"))
    run_margin_sweep(f1_dir=a.out_dir, y=y_val, out_dir=os.path.join(a.out_dir, "margin_sweep"))


if __name__ == "__main__":
    main()
