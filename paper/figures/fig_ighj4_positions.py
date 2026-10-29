"""Supplementary Figure 6: where the IGHJ4 latents 463, 4720 and 6276 of the expansion-32 TopK SAE on p-IgGen layer 3
fire in the 10,269 IGHJ4 validation heavy chains, (a) by token position and (b) by IMGT position.

"Activation" here is a hit: the latent is among the token's top 32 (a count, not a magnitude). Site and encoding are
those of the Section 4.1 gate (paper/scripts/identification/section41_encode.sh): gpt_neox.layers[3] output before the
final layer norm, input "1" + VH + "2" padded to 350 tokens, no attention mask. IMGT labels come from the heavy half of
the paired alignment of datasets/master_Dec13_10.31.csv, whose validation pairs with an ANARCI IGHJ4 call form the set.

Stages (state in <work_dir>/figures/ighj4_positions unless --dir is given)
  table    master_Dec13_10.31.csv INNER JOIN vj_val_anarci.csv on VH+VL, IGHJ4 rows -> ighj4_val.csv.gz   (CPU, ~1 min)
  encode   top-32 latent indices of every token -> hits_pre_vh.npz (row, position) of the three latents   (GPU, ~10 min)
  plot     OUT/absolute_positional_activations_valonly_preln.png (a), OUT/high_res_IMGT_activations_valonly_preln.png
           (b) and OUT/stats_valonly_preln.json (share of each latent's hits at its top IMGT position)     (CPU, ~1 min)
           Reads the state directory, else the deposited figures/ighj4_positions of <results_dir>.

Usage: python -m paper.figures.fig_ighj4_positions {table,encode,plot} [--out OUT] [--dir DIR]
       (paper/scripts/figures/ighj4_positions.sh runs all three; paper/slurm/figures/ighj4_positions.slurm on a GPU node)"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from plm_sae import models, paths as P
from plm_sae import sae as S
from paper.analyses.identification.imgt import CANONICAL_NUMBERING
from paper.benchmark._inputs import inp

LATENTS = [463, 4720, 6276]
HIGHLIGHT = {463: ("120 ", "red", "Position 120"),
             4720: ("119 ", "green", "Position 119"),
             6276: ("117 ", "blue", "Position 117")}
L = 350

# ----------------------------------------------------------------------------- table
def build_table(d):
    master = pd.read_csv(P.dataset("master_Dec13_10.31.csv"), usecols=["VH+VL", "aligned_VHorVHH+VL"])
    val = pd.read_csv(P.dataset("vj_val_anarci.csv"), usecols=["VH+VL", "VHorVHH", "j_gene_anarci_heavy"])
    tab = master.merge(val, on="VH+VL", how="inner")
    tab = tab[tab["j_gene_anarci_heavy"].str.contains("IGHJ4", na=False)].reset_index(drop=True)
    assert tab["VH+VL"].is_unique
    tab.to_csv(d / "ighj4_val.csv.gz", index=False)
    print(f"IGHJ4 validation pairs: {len(tab)} (paper: 10,269) -> {d / 'ighj4_val.csv.gz'}")


# ----------------------------------------------------------------------------- encode
def encode(d, bs=16):
    import torch
    out = d / "hits_pre_vh.npz"
    if out.exists():
        print(f"exists, skip: {out}"); return
    seqs = pd.read_csv(d / "ighj4_val.csv.gz", usecols=["VHorVHH"])["VHorVHH"].tolist()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = models.load("piggen", dev)
    sae = S.load("piggen_topk_r32_layer3")
    We, be, bd = [t.detach().to(dev).float() for t in (sae.W_enc, sae.b_enc, sae.b_dec)]
    cap = []
    model.gpt_neox.layers[3].register_forward_hook(
        lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach().float()))
    idx = np.zeros((len(seqs), L, 32), dtype=np.int16)
    for s in range(0, len(seqs), bs):
        texts = ["1" + q + "2" for q in seqs[s:s + bs]]
        ids = tok(texts, padding="max_length", truncation=True, max_length=L, return_tensors="pt")["input_ids"].to(dev)
        cap.clear()
        with torch.no_grad():
            model(input_ids=ids)
            _, i = torch.relu((cap[0] - bd) @ We.T + be).topk(32, dim=-1)
        idx[s:s + len(texts)] = i.cpu().numpy().astype(np.int16)
        if (s // bs) % 100 == 0:
            print(f"[{s}/{len(seqs)}]", flush=True)
    save = {"n_rows": np.array(len(seqs))}
    for lat in LATENTS:
        sq, ps, _ = np.where(idx == lat)
        save[f"seq_{lat}"] = sq.astype(np.int32); save[f"pos_{lat}"] = ps.astype(np.int16)
        print(f"latent {lat}: {len(sq)} hits", flush=True)
    tmp = str(out) + ".tmp.npz"
    np.savez_compressed(tmp, **save); os.replace(tmp, out)


# ----------------------------------------------------------------------------- plot
def get_canonical_numbers(aligned_seq, canonical_list):
    N, Ls = len(canonical_list), len(aligned_seq)
    if Ls == N:
        chunks = [(aligned_seq, canonical_list)]
    elif Ls == 2 * N:
        chunks = [(aligned_seq[:N], canonical_list), (aligned_seq[N:], canonical_list)]
    else:
        raise ValueError(f"Unexpected aligned length {Ls!r}")
    return [canon[i] for chunk, canon in chunks for i, aa in enumerate(chunk) if aa != '-']


def build_full_canonical_map(aligned_seq, canonical_list, target_len=L):
    """Padded token position -> IMGT label (None for '1', '2' and padding)."""
    try:
        labels = get_canonical_numbers(aligned_seq, canonical_list)
    except Exception:
        return [None] * target_len
    full = [None] * target_len
    raw = ''.join(c for c in aligned_seq if c != '-')
    padded = ('1' + raw + '2')
    padded = padded + '-' * (target_len - len(padded)) if len(padded) < target_len else padded[:target_len]
    ci = 0
    for i, aa in enumerate(padded):
        if aa not in ('-', '1', '2') and ci < len(labels):
            full[i] = labels[ci]
            ci += 1
    return full


def pad(seq):
    s = '1' + seq + '2'
    return s + '-' * (L - len(s)) if len(s) < L else s[:L]


def plot(d, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    tab = pd.read_csv(d / "ighj4_val.csv.gz")
    h = np.load(d / "hits_pre_vh.npz")
    assert int(h['n_rows']) == len(tab), (int(h['n_rows']), len(tab))
    raw_seqs = tab['VHorVHH'].values
    aligned = np.array([al[:200] for al in tab['aligned_VHorVHH+VL'].values], dtype=object)   # heavy half
    stats = {'n_sequences': int(len(tab)), 'latents': {}}
    abs_data, imgt_data, cmaps = {}, {}, {}
    for v in LATENTS:
        s, p = h[f'seq_{v}'], h[f'pos_{v}'].astype(int)
        # (a) share of all hits of this latent at each padded position
        pc = Counter(p.tolist())
        tot = len(p)
        abs_data[v] = (sorted(pc), [pc[q] / tot * 100 for q in sorted(pc)])
        # (b) hits mapped to IMGT labels; start, end, padding and unmapped tokens skipped
        pairs, skipped = [], 0
        for si, pi in zip(s.tolist(), p.tolist()):
            if si not in cmaps:
                cmaps[si] = build_full_canonical_map(aligned[si], CANONICAL_NUMBERING)
            cp = cmaps[si][pi]
            if cp is None:
                skipped += 1
                continue
            aa = pad(raw_seqs[si])[pi]
            if aa in ('-', '1', '2'):
                skipped += 1
                continue
            pairs.append((aa, cp))
        cc = Counter(c for _, c in pairs)
        ac = Counter(pairs)
        imgt_data[v] = (cc, len(pairs), tot)
        lab = HIGHLIGHT[v][0]
        top_lab, top_ct = cc.most_common(1)[0]
        stats['latents'][str(v)] = {
            'total_hits': int(tot), 'valid_mapped_hits': len(pairs), 'skipped': skipped,
            'unique_sequences': int(len(np.unique(s))),
            'top_abs_position': int(max(pc, key=pc.get)),
            'top_abs_pct': round(max(pc.values()) / tot * 100, 2),
            'highlight_label': lab.strip(),
            'highlight_pct_of_valid': round(cc[lab] / len(pairs) * 100, 2),
            'highlight_pct_of_total': round(cc[lab] / tot * 100, 2),
            'top_imgt_label': top_lab.strip(), 'top_imgt_pct_of_valid': round(top_ct / len(pairs) * 100, 2),
            'top_aa_imgt': [[a, c.strip(), n, round(n / len(pairs) * 100, 2)] for (a, c), n in ac.most_common(5)],
        }
    print(json.dumps({k: {kk: vv for kk, vv in x.items() if kk != 'top_aa_imgt'} for k, x in stats['latents'].items()}, indent=1))
    with open(os.path.join(out, "stats_valonly_preln.json"), 'w') as f:
        json.dump(stats, f, indent=1)

    rc = {'font.size': 20, 'axes.titlesize': 24, 'axes.labelsize': 22,
          'xtick.labelsize': 22, 'ytick.labelsize': 22, 'legend.fontsize': 22}

    # (a) token positions, 1196 x 1600 px RGBA
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(11.96, 16), dpi=100)
        for i, v in enumerate(LATENTS):
            ax = plt.subplot(3, 1, i + 1)
            x, y = abs_data[v]
            ax.bar(x, y, alpha=0.7)
            ax.set_title(f"Activation positions for latent {v}")
            ax.set_xlabel("Sequence position")
            ax.set_ylabel("Percentage of activations (%)")
            ax.set_xlim(0, 350)
            ticks = list(range(0, 350, 25))
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks)
        fig.canvas.draw()
        plt.tight_layout(pad=0.3, h_pad=1.08)
        fig.canvas.draw()
        plt.tight_layout(pad=0.3, h_pad=1.08)  # second pass: tick-label widths settle after a draw
        fa = os.path.join(out, 'absolute_positional_activations_valonly_preln.png')
        fig.savefig(fa, dpi=100)
        plt.close(fig)

    # (b) IMGT positions, 1188 x 1600 px RGB
    idx_map = {lab: i for i, lab in enumerate(CANONICAL_NUMBERING)}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(11.88, 16), dpi=100)
        for i, v in enumerate(LATENTS, start=1):
            ax = plt.subplot(3, 1, i)
            cc, nvalid, tot = imgt_data[v]
            labs = list(cc.keys())
            ax.bar([idx_map[l] for l in labs], [cc[l] / nvalid * 100 for l in labs],
                   alpha=0.7, color='skyblue')
            lab, col, leg = HIGHLIGHT[v]
            ax.bar([idx_map[lab]], [cc[lab] / nvalid * 100], color=col, label=leg)
            ax.legend()
            step = max(1, len(CANONICAL_NUMBERING) // 14)
            tl = list(range(0, len(CANONICAL_NUMBERING), step))
            ax.set_xticks(tl)
            ax.set_xticklabels([CANONICAL_NUMBERING[j] for j in tl], fontsize=20)
            ax.set_xlim(-0.5, len(CANONICAL_NUMBERING) - 0.5)
            ax.set_title(f"Activation positions for latent {v}")
            ax.set_ylabel("Percentage of activations (%)")
            ax.set_xlabel("IMGT position")
        fig.canvas.draw()
        plt.tight_layout(pad=0.3, h_pad=1.08)
        fig.canvas.draw()
        plt.tight_layout(pad=0.3, h_pad=1.08)
        fb = os.path.join(out, 'high_res_IMGT_activations_valonly_preln.png')
        fig.savefig(fb, dpi=100)
        plt.close(fig)

    for fn, mode in [(fa, 'RGBA'), (fb, 'RGB')]:   # same pixel size and mode as the manuscript files, no DPI chunk
        im = Image.open(fn).convert(mode)
        im.save(fn)
        print("Supplementary Figure 6:", fn, im.size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["table", "encode", "plot"])
    ap.add_argument("--dir", default=None, help="state directory (default <work_dir>/figures/ighj4_positions)")
    ap.add_argument("--out", default=".", help="figure directory (plot)")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args(argv)
    if a.stage == "plot":
        os.makedirs(a.out, exist_ok=True)
        plot(Path(a.dir) if a.dir else inp("figures", "ighj4_positions", "hits_pre_vh.npz").parent, a.out)
        return
    d = Path(a.dir) if a.dir else P.work("figures", "ighj4_positions")
    d.mkdir(parents=True, exist_ok=True)
    build_table(d) if a.stage == "table" else encode(d, a.bs)


if __name__ == "__main__":
    main()
