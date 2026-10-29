"""Steered generation for p-IgGen, IgLM and ProGen2-OAS: many (latent, alpha) configurations in one batched sampling
loop, each batch row with its own steering vector.

The vector is alpha times the latent's unit decoder vector, added at the SAE's site to every position from START on
(and before END, if given):
  --site post       output of the final layer norm (the hyperparameter-matched SAEs)
  --site blockout   output of block --layer (the expansion-32 TopK SAE of Section 4.1)
Sampling is each model's own: p-IgGen T 1.25, top-p 0.95, from "1"; IgLM T 1.0 from [HEAVY][HUMAN]EVQ, single-token
bad words and forced EOS as IgLM's generate; ProGen2-OAS T 0.2, top-p 0.95, from "1EVQ" (--temp, --top_p and --prompt
override; the paper's ProGen2-OAS benchmark used --temp 1.0 --prompt 1EVQLVESGGGLVQPGGSLRLSCAAS --tf32). Invalid
outputs are discarded (and IgLM duplicates, as IgLM does) until each configuration has --total sequences.

--configs: one line per configuration, "LATENT ALPHA [START [TAG [END]]]"; TAG is the output subdirectory.
Output: OUTDIR[/TAG]/steered_l{layer}_f{latent}_a{alpha:.2f}_seed{s}_seq{total}.txt, one sequence per line.

Reproducibility: rows of the same configuration index share random numbers across configurations (common random
numbers), so a configuration's library depends on the whole --configs file and on --rows (by default derived from
GPU memory); libraries are byte-identical only for the same configs file on the same GPU type.

Run: python -m plm_sae.steering.generate --model MODEL --sae SAE --configs CFG --outdir DIR
     [--site post] [--layer 3] [--seeds 0 1 2] [--total 1000] [--min_yield 0.08] [--tf32] [--temp T --top_p P --prompt S]
(scripts/steering/steer.sh writes the configs file and runs it.)"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from plm_sae import models, paths as P, sae as S
from plm_sae.steering import decode as D

ROWS_PER_GB = {"piggen": 0.012, "IgLM": 0.004, "progen2_oas": 0.24}


def progen2_truncate(text, terminals=("1", "2")):
    """Cut a ProGen2 sample at its first terminal token after the start."""
    pos = [p for p in (text.find(t, 1) for t in terminals) if p != -1]
    return text[:min(pos) + 1] if pos else text


def load_model(name, device, prompt=None):
    """(model, gen(B, temp, top_p, steerer, crn) -> token ids, post(ids) -> sequences or None, dedupe)."""
    lm, tok = models.load(name, device)
    if name == "piggen":
        from piggen.utils import format_and_validate_output
        ctx = tok.encode("1", return_tensors="pt").to(device)
        eos = tok.encode("2")[0]

        def gen(B, temp=1.25, top_p=0.95, st=None, crn=None):
            return D.sample(lm, ctx, B, temp, top_p, 400, [eos], tok.pad_token_id, steerer=st, crn=crn)

        def post(ids):
            return [format_and_validate_output(s) for s in tok.batch_decode(ids, skip_special_tokens=True)]
        return lm, gen, post, False
    if name == "IgLM":
        from iglm.model.tokens import BAD_WORD_IDS
        from iglm.model.utils import iglm_to_infilled, validate_generated_seq
        ctx = torch.tensor([tok.convert_tokens_to_ids(["[HEAVY]", "[HUMAN]"] + list("EVQ"))], device=device)
        bad = [b[0] for b in BAD_WORD_IDS]

        def gen(B, temp=1.0, top_p=1.0, st=None, crn=None):
            return D.sample(lm, ctx, B, temp, top_p, 150 - ctx.shape[1], [tok.cls_token_id], tok.pad_token_id,
                            steerer=st, crn=crn, bad_ids=bad, forced_eos=tok.cls_token_id)

        def post(ids):
            return ["".join(tok.decode(iglm_to_infilled(s, tok))).replace(" ", "") if validate_generated_seq(s, tok)
                    else None for s in ids.cpu().numpy()]
        return lm, gen, post, True
    if name == "progen2_oas":
        from tokenizers import Tokenizer
        rt = Tokenizer.from_str((P.progen2_dir() / "tokenizer.json").read_text())
        pad = rt.encode("<|pad|>").ids[0]
        ctx = torch.tensor(rt.encode(prompt or "1EVQ").ids, device=device).unsqueeze(0)
        term = [rt.token_to_id("1"), rt.token_to_id("2")]

        def gen(B, temp=0.2, top_p=0.95, st=None, crn=None):
            return D.sample(lm, ctx, B, temp, top_p, 256 - ctx.shape[1], term, pad, steerer=st, crn=crn)

        def post(ids):
            return [progen2_truncate(s).replace("1", "").replace("2", "") for s in rt.decode_batch(ids.tolist())]
        return lm, gen, post, False
    raise ValueError(name)


class RowSteerer:
    """Adds V[row] at the site. With S set, row b is steered only at positions >= S[b]; with E set, also only at
    positions < E[b] where E[b] > 0. p0 is the absolute position of the first token of the current forward pass
    (set by decode.sample)."""

    def __init__(self, lm, site, layer):
        core = lm.gpt_neox if hasattr(lm, "gpt_neox") else lm.transformer
        blocks = core.layers if hasattr(core, "layers") else core.h
        lnf = core.final_layer_norm if hasattr(core, "final_layer_norm") else core.ln_f
        self.V = self.S = self.E = None
        self.p0 = 0
        if site == "post":
            lnf.register_forward_hook(lambda mod, i, o: o + self.vec(o))
        elif site == "blockout":
            def hook(mod, i, o):
                return (o[0] + self.vec(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else o + self.vec(o)
            blocks[layer].register_forward_hook(hook)
        else:
            raise ValueError(site)

    def vec(self, h):
        v = self.V[:, None, :].to(h.dtype)
        if self.S is None:
            return v
        pos = self.p0 + torch.arange(h.shape[1], device=h.device)
        on = pos[None, :] >= self.S[:, None]
        if self.E is not None:
            on = on & ((self.E[:, None] <= 0) | (pos[None, :] < self.E[:, None]))
        return v * on.unsqueeze(-1).to(h.dtype)


def read_configs(path):
    """(latent, alpha, start, tag, end) per line; start 0 = from the first position, end 0 = to the end."""
    out = []
    for line in open(path):
        f = line.split()
        if not f:
            continue
        tag = f[3] if len(f) > 3 else ("" if len(f) <= 2 else "gated")
        out.append((f[0], float(f[1]), int(f[2]) if len(f) > 2 else 0, tag, int(f[4]) if len(f) > 4 else 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=models.MODELS)
    ap.add_argument("--sae", required=True, help="SAE (or direction file) whose decoder vectors steer")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--site", default="post", choices=["post", "blockout"])
    ap.add_argument("--layer", type=int, default=3, help="block of --site blockout; also the l<layer> in file names")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--rows", type=int, default=0, help="rows per sampling call (0: from GPU memory)")
    ap.add_argument("--max_rounds", type=int, default=10)
    ap.add_argument("--min_yield", type=float, default=0.0,
                    help="abandon a configuration once a round yields fewer new valid sequences than this fraction of "
                         "its rows (the paper: 0.08; such a configuration cannot reach 80%% coverage in 10 rounds)")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matrix products (the paper's ProGen2-OAS runs)")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--prompt", default=None, help="ProGen2-OAS only; must start with '1'")
    a = ap.parse_args()
    if a.prompt:
        assert a.model == "progen2_oas" and a.prompt.startswith("1"), "--prompt is for ProGen2-OAS and starts with '1'"
    samp = {k: v for k, v in (("temp", a.temp), ("top_p", a.top_p)) if v is not None}
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    lm, gen, post, dedupe = load_model(a.model, dev, a.prompt)
    W = S.decoder(a.sae)
    assert W.shape[1] == models.D_MODEL[a.model], (W.shape, a.model)
    cfgs = read_configs(a.configs)
    vec = {c: (c[1] * W[int(c[0])] / W[int(c[0])].norm()).to(dev) for c in cfgs}
    gated, ended = any(c[2] > 0 or c[4] > 0 for c in cfgs), any(c[4] > 0 for c in cfgs)
    if not a.rows:
        gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        a.rows = int(max(32, min(4096, (gb - 6) / ROWS_PER_GB[a.model])))
    st = RowSteerer(lm, a.site, a.layer)
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    for seed in a.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        got = {c: [] for c in cfgs}
        seen = {c: set() for c in cfgs}
        abandoned = set()
        for rnd in range(a.max_rounds):
            need = {c: a.total - len(got[c]) for c in cfgs if len(got[c]) < a.total and c not in abandoned}
            if not need:
                break
            # row j of every configuration sits in the same call, so rows with the same j share random numbers
            n_rows = {c: int(math.ceil(k * (1.15 if rnd == 0 else 1.5)) + 2) for c, k in need.items()}
            rows = [(c, j) for j in range(max(n_rows.values())) for c in n_rows if j < n_rows[c]]
            before = {c: len(got[c]) for c in n_rows}
            step = max(1, a.rows // len(n_rows)) * len(n_rows)
            for b in range(0, len(rows), step):
                cj = rows[b:b + step]
                chunk = [c for c, _ in cj]
                st.V = torch.stack([vec[c] for c in chunk])
                st.S = torch.tensor([c[2] for c in chunk], device=dev) if gated else None
                st.E = torch.tensor([c[4] for c in chunk], device=dev) if ended else None
                js = torch.tensor([j for _, j in cj])
                with torch.no_grad():
                    ids = gen(len(chunk), st=st, crn=js - js.min(), **samp)
                for c, s in zip(chunk, post(ids)):
                    if s is None or len(got[c]) >= a.total:
                        continue
                    if dedupe:
                        if s in seen[c]:
                            continue
                        seen[c].add(s)
                    got[c].append(s)
            if a.min_yield > 0:
                abandoned |= {c for c, k in n_rows.items()
                              if len(got[c]) < a.total and (len(got[c]) - before[c]) / k < a.min_yield}
        short = {f"{c[0]}@{c[1]}@{c[2]}": len(v) for c, v in got.items() if len(v) < a.total}
        if short:
            print(f"WARNING seed {seed}: {len(short)} configurations short of {a.total}: {short}", flush=True)
        for (k, al, _, tag, _), v in got.items():
            od = os.path.join(a.outdir, tag)
            os.makedirs(od, exist_ok=True)
            with open(f"{od}/steered_l{a.layer}_f{k}_a{al:.2f}_seed{seed}_seq{a.total}.txt", "w") as f:
                f.write("\n".join(v) + ("\n" if v else ""))
        print(f"seed {seed} done, {time.time() - t0:.0f}s", flush=True)
    json.dump({"model": a.model, "site": a.site, "rows": a.rows, "tf32": a.tf32, "seeds": a.seeds,
               "n_configs": len(cfgs), "min_yield": a.min_yield, "sampling": samp, "prompt": a.prompt,
               "seconds": round(time.time() - t0, 1)}, open(f"{a.outdir}/generate_log.json", "w"))


if __name__ == "__main__":
    main()
