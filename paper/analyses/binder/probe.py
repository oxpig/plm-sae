"""Binder status on the hyperparameter-matched p-IgGen SAEs (Supplementary Section A.11): a logistic-regression probe
over the whole dictionary and the per-latent F1 gate on the binder label, for the TopK (piggen_topk_r8) and Ordered
(piggen_ordered_r8) SAEs.

Latents: <work_dir>/activations/<sae>/val/, per-token latents of the 30,754 validation sequences (as for the J-gene
gates), each file mean-pooled over its 350 positions. Labels: binder label of <data_dir>/datasets/rbd_binding_master.csv
joined on VH+VL (1,324 binders, 4.3%).
Probe: MinMax scaling, 5-fold stratified CV (shuffled, seed 0), L2 logistic regression (liblinear, balanced class
weights) with an inner 3-fold grid over C in 10^-3..10^2; ROC AUC and PR-AUC over the folds.
Gate: per-latent best F1 over thresholds 0.1-1.0 for the 500 latents with the largest |mean probe coefficient|; a
latent is a feature iff F1 > F1_all+ + margin (margins 0.10-0.25). The whole dictionary is also swept (thresholds
0.1-1.0 with 0.9), so no latent outside the shortlist is missed.

run     python -m paper.analyses.binder.probe run SAE [--ncpu 16]
        writes <work_dir>/binder/<sae>/{summary.json, probe_summary.json, f1_scores_class_binder.csv,
        f1_all_latents.csv, probe_coef.npy, oof.npy} (CPU, 16 cores, 64 GB, about 6 min per SAE)
report  python -m paper.analyses.binder.probe report [--tex manuscript.tex]
        prints the numbers of Supplementary Section A.11 (from <work_dir>, else <results_dir>); with --tex, checks
        that each appears in the binder subsection"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from plm_sae import paths as P
from paper.benchmark._inputs import inp

TOP_K = 500
THRESHOLDS = [0.1, 0.2, 0.5, 0.8, 1.0]
THR_ALL = [0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
MARGINS = [0.10, 0.15, 0.20, 0.25]
SAES = {"TopK": "piggen_topk_r8", "Ordered": "piggen_ordered_r8"}


def load_latent_matrix(act_dir):
    """(orig_rows, Z) with Z[i] the mean over positions of the file of orig_rows[i], in orig_row order (the CV folds
    depend on the row order)."""
    files = glob.glob(os.path.join(act_dir, "*.npy"))
    if not files:
        raise SystemExit(f"no .npy files in {act_dir}")
    row = lambda f: int(os.path.basename(f).split(".")[0].split("_")[0])
    files = sorted(files, key=row)
    rows = np.array([row(f) for f in files])
    Z = np.stack([np.load(f).mean(axis=0) for f in files]).astype(np.float32)
    return rows, Z


def align_labels(rows, val_csv, master_csv):
    """Binder label (float) of each validation row, joined on VH+VL (first label of a repeated pair), and a matched mask."""
    val = pd.read_csv(val_csv, usecols=["VH+VL"])
    val["row"] = val.index
    master = pd.read_csv(master_csv, usecols=["VH+VL", "label"]).drop_duplicates(subset=["VH+VL"], keep="first")
    m = val.merge(master, on="VH+VL", how="left").drop_duplicates(subset=["row"], keep="first").set_index("row").loc[rows]
    return m["label"].astype(float).to_numpy(), m["label"].notna().to_numpy()


def train_probe(X, y, cpus=None, cv=5, seed=0):
    """Out-of-fold probabilities, mean coefficient over folds, per-fold ROC AUC and PR-AUC, last fold's C."""
    Xs = MinMaxScaler().fit_transform(X)
    oof, coefs, auc, pr_auc = np.zeros(len(y)), [], [], []
    for tr, te in StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed).split(Xs, y):
        gs = GridSearchCV(LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced"),
                          {"C": np.logspace(-3, 2, 6)}, cv=3, scoring="roc_auc", n_jobs=cpus)
        gs.fit(Xs[tr], y[tr])
        oof[te] = gs.predict_proba(Xs[te])[:, 1]
        coefs.append(gs.best_estimator_.coef_[0])
        auc.append(roc_auc_score(y[te], oof[te]))
        pr_auc.append(average_precision_score(y[te], oof[te]))
    return dict(oof=oof, coef=np.mean(coefs, axis=0), auc=auc, pr_auc=pr_auc, best_C=gs.best_params_["C"])


def f1_all_plus(y):
    """F1 of predicting every sequence a binder."""
    return 2.0 * y.mean() / (1.0 + y.mean())


def gate(Z, y, coef):
    """Top TOP_K latents by |coef|; per-latent best F1 over THRESHOLDS on MinMax-scaled activations."""
    top = np.argsort(np.abs(coef))[::-1][:TOP_K]
    Zs = MinMaxScaler().fit_transform(Z[:, top])
    recs = []
    for j, feat in enumerate(top):
        vals = np.clip(Zs[:, j].astype(np.float32), 0.0, 1.0)
        best_f1, best_thr = 0.0, None
        for thr in THRESHOLDS:
            f1 = f1_score(y, vals >= thr, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        recs.append(dict(feature_idx=int(feat), weight=float(coef[feat]), best_threshold=best_thr, best_f1=best_f1,
                         average_precision=average_precision_score(y, vals)))
    df = pd.DataFrame(recs).sort_values("best_f1", ascending=False).reset_index(drop=True)
    return df, {f"{m:.2f}": int((df.best_f1 > f1_all_plus(y) + m).sum()) for m in MARGINS}


def gate_all(Z, y):
    """Whole-dictionary sweep (F1 = 2TP / (predicted positives + positives))."""
    Zs = np.clip(MinMaxScaler().fit_transform(Z).astype(np.float32), 0.0, 1.0)
    pos = y == 1
    best, bthr = np.zeros(Z.shape[1]), np.full(Z.shape[1], np.nan)
    for thr in THR_ALL:
        pred = Zs >= thr
        tp, pp = pred[pos].sum(0), pred.sum(0)
        f1 = np.where(pp + pos.sum() > 0, 2 * tp / np.maximum(pp + pos.sum(), 1), 0.0)
        upd = f1 > best
        best[upd], bthr[upd] = f1[upd], thr
    df = pd.DataFrame(dict(feature_idx=np.arange(Z.shape[1]), best_threshold=bthr, best_f1=best,
                           ever_nonzero=(Z > 0).any(0))).sort_values("best_f1", ascending=False).reset_index(drop=True)
    return df, {f"{m:.2f}": int((df.best_f1 > f1_all_plus(y) + m).sum()) for m in MARGINS}


def run(sae, ncpu):
    out = P.work("binder", sae)
    os.makedirs(out, exist_ok=True)
    if (out / "summary.json").exists():
        print("exists, skip:", out / "summary.json")
        return
    rows, Z = load_latent_matrix(P.activations(sae, "val"))
    y, matched = align_labels(rows, P.dataset("vj_val_anarci.csv"), P.dataset("rbd_binding_master.csv"))
    assert matched.all(), f"{(~matched).sum()} sequences without a binder label"
    print(f"Z {Z.shape}, binders {int(y.sum())} ({100 * y.mean():.2f}%)", flush=True)
    summ = dict(checkpoint=sae, n=int(len(y)), n_binder=int(y.sum()))
    if (out / "probe_coef.npy").exists():   # resume after a requeue: the probe is the slow part
        coef = np.load(out / "probe_coef.npy")
        summ.update(json.load(open(out / "probe_summary.json")))
    else:
        t0 = time.time()
        r = train_probe(Z, y, cpus=ncpu)
        pr = dict(auc_mean=float(np.mean(r["auc"])), auc_std=float(np.std(r["auc"])),
                  pr_auc_mean=float(np.mean(r["pr_auc"])), pr_auc_std=float(np.std(r["pr_auc"])),
                  best_C_last_fold=float(r["best_C"]), auc_folds=[float(v) for v in r["auc"]],
                  pr_auc_folds=[float(v) for v in r["pr_auc"]], probe_seconds=time.time() - t0)
        coef = r["coef"]
        np.save(out / "oof.npy", r["oof"])
        json.dump(pr, open(out / "probe_summary.json", "w"), indent=1)
        np.save(str(out / "probe_coef.npy") + ".tmp.npy", coef)
        os.replace(str(out / "probe_coef.npy") + ".tmp.npy", out / "probe_coef.npy")
        summ.update(pr)
    yi = y.astype(int)
    df, counts = gate(Z, yi, coef)
    df.to_csv(out / "f1_scores_class_binder.csv", index=False)
    summ.update(f1_all_plus=f1_all_plus(yi), best_latent=int(df.feature_idx[0]), best_f1=float(df.best_f1[0]),
                best_threshold=df.best_threshold[0], n_features_by_margin_top500=counts)
    da, ca = gate_all(Z, yi)
    da.to_csv(out / "f1_all_latents.csv", index=False)
    summ.update(all_latents_best_latent=int(da.feature_idx[0]), all_latents_best_f1=float(da.best_f1[0]),
                all_latents_n_features_by_margin=ca)
    json.dump(summ, open(str(out / "summary.json") + ".tmp", "w"), indent=1, default=float)
    os.replace(str(out / "summary.json") + ".tmp", out / "summary.json")
    print(json.dumps(summ, indent=1, default=float))


def report(tex):
    vals = []
    for arch, sae in SAES.items():
        s = json.load(open(inp("binder", sae, "summary.json")))
        feats = max(s["n_features_by_margin_top500"].values()) + max(s["all_latents_n_features_by_margin"].values())
        print(f"{arch}: ROC AUC {s['auc_mean']:.3f} +- {s['auc_std']:.3f}, PR-AUC {s['pr_auc_mean']:.3f} +- {s['pr_auc_std']:.3f}, "
              f"best latent of the whole dictionary {s['all_latents_best_latent']} F1 {s['all_latents_best_f1']:.3f} "
              f"(of the probe shortlist: {s['best_latent']}, {s['best_f1']:.3f}), F1_all+ {s['f1_all_plus']:.3f}, "
              f"binders {s['n_binder']}/{s['n']}, features at any margin {feats}")
        vals += [f"{s['auc_mean']:.3f} \\pm {s['auc_std']:.3f}", f"{s['pr_auc_mean']:.3f}", f"{s['pr_auc_std']:.3f}",
                 f"{s['all_latents_best_f1']:.3f}", f"{s['f1_all_plus']:.3f}"]
        assert feats == 0, "the paper reports no binder feature"
    if not tex:
        return 0
    t = open(tex).read()
    i = t.index("\\label{supp:binder}")
    body = t[i:t.index("\\subsection", i + 1)]
    miss = [v for v in vals if v not in body]
    print(f"Supplementary A.11 binder numbers: {len(vals) - len(miss)}/{len(vals)} found in the text", *(["missing:"] + miss if miss else []))
    return len(miss)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("sae", nargs="?", choices=list(SAES.values()))
    ap.add_argument("--ncpu", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    ap.add_argument("--tex")
    a = ap.parse_args()
    if a.mode == "run":
        if not a.sae:
            ap.error("run needs SAE")
        run(a.sae, a.ncpu)
    else:
        sys.exit(1 if report(a.tex) else 0)


if __name__ == "__main__":
    main()
