"""AbLang-2 pseudo-log-likelihood (ablang2-paired, installed `ablang2` package) of the sequences of the steered generation
analysis: every library of <work_dir>/quality/list.csv (its out/<tag>/steered_aligned.csv.gz, as the other metrics) and
the natural references (the first 1,000 paired OAS training sequences; paired for p-IgGen, heavy chains for IgLM and
ProGen2-OAS). Each residue is masked in turn and the log-probability of the true residue is averaged over the residues;
heavy chains alone are given as "<VH>|". Sequences with a non-standard residue or a heavy chain shorter than 50
residues get NaN. Several sequences share one forward pass (AbLang-2 masks padding); fp16 autocast with --fp16.
Writes <work_dir>/quality/ablang/scores/<tag>.npy (one value per row of steered_aligned.csv.gz). Array tasks share the
work through lock files; finished tags are skipped. MAXROWS (environment, default 1536) bounds the masked copies per
forward pass. `check` compares the batched scorer with AbLang-2's own pseudo_log_likelihood on 16 natural pairs.
Usage: "$PLM_SAE_PYTHON_ABLANG" -m paper.analyses.quality.ablang_scores run|check [--fp16]"""
import argparse
import os, time
import numpy as np, pandas as pd
import torch
from plm_sae import paths as P

Q = P.work("quality")
O = Q / "ablang"
FP16 = False
AA = set("ACDEFGHIKLMNPQRSTVWY")
MAXROWS = int(os.environ.get("MAXROWS", 1536))


def jobs():
    """(tag, kind, path) of every library of list.csv and the two natural references."""
    L = pd.read_csv(Q / "list.csv")
    out = [(t, "lib", Q / "out" / t / "steered_aligned.csv.gz") for t in L.tag]
    out += [("natural_paired", "natural", None), ("natural_heavy", "natural", None)]
    return [j for j in out if j[2] is None or j[2].exists()]


def sequences(tag, kind, path):
    if kind == "natural":
        T = pd.read_csv(P.dataset("paired_oas_human_train_aligned.csv")).iloc[:1000]
        vh = T.sequence_alignment_aa_heavy.astype(str).tolist()
        vl = T.sequence_alignment_aa_light.astype(str).tolist() if tag == "natural_paired" else [""] * len(vh)
        return vh, vl
    S = pd.read_csv(path); vh = S.VH.astype(str).tolist()
    vl = S.VL.fillna("").astype(str).tolist() if "VL" in S else [""] * len(vh)
    return vh, [x if x != "nan" else "" for x in vl]


class Scorer:
    def __init__(self):
        import ablang2
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.M = ablang2.pretrained(model_to_use="ablang2-paired", device=self.dev); self.M.freeze(); self.tok = self.M.tokenizer
        self.special = torch.tensor(self.tok.all_special_tokens, device=self.dev)

    def encode(self, vh, vl):
        return self.tok([f"<{vh}>|<{vl}>" if vl else f"<{vh}>|"], pad=True, w_extra_tkns=False, device=self.dev)[0]

    def pll_many(self, vhs, vls):
        """Mean masked log-probability per sequence; sequences batched so that each forward has <= MAXROWS masked copies."""
        out = np.full(len(vhs), np.nan); todo = [i for i in range(len(vhs)) if set(vhs[i] + vls[i]) <= AA and len(vhs[i]) >= 50]
        k = 0
        while k < len(todo):
            grp, rows, tmax = [], 0, 0
            while k < len(todo):   # budget rows x length^2 (attention), so long degenerate sequences do not run out of memory
                lab = self.encode(vhs[todo[k]], vls[todo[k]]); n = int((~torch.isin(lab, self.special)).sum())
                if grp and (rows + n) * max(tmax, len(lab)) ** 2 > MAXROWS * 240 ** 2: break
                grp.append((todo[k], lab)); rows += n; tmax = max(tmax, len(lab)); k += 1
            T = max(len(l) for _, l in grp); X, tgt, pos, owner = [], [], [], []
            for j, (i, lab) in enumerate(grp):
                idx = (~torch.isin(lab, self.special)).nonzero()[:, 0]
                x = torch.full((len(idx), T), self.tok.pad_token, device=self.dev, dtype=lab.dtype); x[:, :len(lab)] = lab
                x[torch.arange(len(idx), device=self.dev), idx] = self.tok.mask_token
                X.append(x); tgt.append(lab[idx]); pos.append(idx); owner += [j] * len(idx)
            X, tgt, pos = torch.cat(X), torch.cat(tgt), torch.cat(pos)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=FP16 and self.dev == "cuda"):
                lg = self.M.AbLang(X)
            lg = lg[torch.arange(len(X), device=self.dev), pos].float(); lg[:, self.tok.all_special_tokens] = -float("inf")
            lp = torch.log_softmax(lg, -1).gather(1, tgt[:, None])[:, 0].cpu().numpy(); owner = np.array(owner)
            for j, (i, _) in enumerate(grp): out[i] = lp[owner == j].mean()
        return out


def run():
    (O / "scores").mkdir(parents=True, exist_ok=True)
    (O / "locks").mkdir(exist_ok=True)
    sc = None
    for tag, kind, path in jobs():
        f = O / "scores" / f"{tag}.npy"
        if f.exists():
            continue
        try:
            os.close(os.open(O / "locks" / tag, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        sc = sc or Scorer()
        t = time.time()
        v = sc.pll_many(*sequences(tag, kind, path))
        np.save(str(f) + ".tmp.npy", v)
        os.replace(str(f) + ".tmp.npy", f)
        print(tag, len(v), "seqs", int(np.isfinite(v).sum()), "scored", f"mean {np.nanmean(v):.4f}", f"{time.time() - t:.0f}s", flush=True)


def check():
    """Batched scorer vs AbLang-2's own pseudo_log_likelihood (paired and heavy-only), and speed."""
    sc = Scorer(); vh, vl = sequences("natural_paired", "natural", None); vh, vl = vh[:16], vl[:16]
    ref = sc.M([[h, l] for h, l in zip(vh, vl)], mode="pseudo_log_likelihood"); t = time.time(); ours = sc.pll_many(vh, vl); dt = (time.time() - t) / 16
    print("paired  ours", np.round(ours[:6], 4), "ablang2", np.round(np.asarray(ref)[:6], 4), "max |diff|", float(np.max(np.abs(ours - np.asarray(ref)))), f"{dt:.3f} s/seq fp16={FP16}")
    refh = sc.M([[h, ""] for h in vh], mode="pseudo_log_likelihood"); oursh = sc.pll_many(vh, [""] * 16)
    print("heavy   ours", np.round(oursh[:6], 4), "ablang2", np.round(np.asarray(refh)[:6], 4), "max |diff|", float(np.max(np.abs(oursh - np.asarray(refh)))))
    print(len(jobs()), "jobs")


def main():
    global FP16
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["run", "check"])
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    FP16 = a.fp16
    {"run": run, "check": check}[a.mode]()


if __name__ == "__main__":
    main()
