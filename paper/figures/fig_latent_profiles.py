"""Figure 1 (latent 4280 against hidden neuron 45) and Supplementary Figure 5 (IGHJ3 latents 3054, 532 and 3098) of
the expansion-32 TopK SAE on p-IgGen layer 3, for one IGHJ3 heavy chain.

Site and encoding are those of the Section 4.1 gate (paper/scripts/identification/section41_encode.sh): the output of
gpt_neox.layers[3] before the final layer norm, input "1" + VH + "2" padded to 350 tokens, no attention mask,
topk_32(relu((x - b_dec) @ W_enc.T + b_enc)), raw activations. The model is causal, so the VH activations are the
same with or without the light chain. Each series is divided by its largest absolute value over the VH; CDRs are
shaded. Runs on CPU in seconds.

The sequence (given below with its CDRs) is a paired heavy and light chain from the Observed Antibody Space (OAS; Olsen
et al. 2022, Kovaltsuk et al. 2018), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

Inputs: checkpoint piggen_topk_r32_layer3 and the p-IgGen model (plm_sae.models).
Outputs: OUT/combined_latent_neuron_plot_crop_preln.png (Figure 1), OUT/Your_paragraph_text_preln.png
(Supplementary Figure 5), the file names of the manuscript.
Usage: python -m paper.figures.fig_latent_profiles OUT"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from plm_sae import models
from plm_sae import sae as S

VH = "EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYTMTWVRQAPGKGLEWVSSISSVSSYRHYADSVNGRFTISRDNARNSLFLQLNSLRAEDTAVYFCAFELEHDDAFDIWGQGTMVTVSS"
VL = "DIQMTQSPSSLSASVGDRVTITCRASQSISTYLNWYQQKPGRAPKLLIYAASHLHSGVPSRLSGSGSGTDFTLTINSLQPEDFATYYCQQSYRTPYTFGQGTKLEIK"
CDRS = dict(cdr1h="GFTFSSY", cdr2h="SSVSSY", cdr3h="ELEHDDAFDI", cdr1l="RASQSISTYLN", cdr2l="AASHLHS", cdr3l="QQSYRTPYT")


def cdr_mask(vh, vl, cdr1h, cdr2h, cdr3h, cdr1l, cdr2l, cdr3l):
    """One character per residue: A/B/C heavy CDRs, Z/Y/W light CDRs, '!' on the last VH residue."""
    sym = {'cdr1h': 'A', 'cdr2h': 'B', 'cdr3h': 'C', 'cdr1l': 'Z', 'cdr2l': 'Y', 'cdr3l': 'W'}
    mask = ['-'] * len(vh + vl)

    def rep(seq, cdr, off, s):
        st = seq.find(cdr)
        if st != -1:
            for i in range(st + off, st + off + len(cdr)):
                mask[i] = s
    rep(vh, cdr1h, 0, sym['cdr1h']); rep(vh, cdr2h, 0, sym['cdr2h']); rep(vh, cdr3h, 0, sym['cdr3h'])
    rep(vl, cdr1l, len(vh), sym['cdr1l']); rep(vl, cdr2l, len(vh), sym['cdr2l']); rep(vl, cdr3l, len(vh), sym['cdr3l'])
    if vh:
        mask[len(vh) - 1] = '!'
    return ''.join(mask)


def activations():
    """Layer-3 pre-LN activations [350, 768] of '1' + VH + '2' and the SAE encoder."""
    model, tok = models.load("piggen")
    cap = {}
    model.gpt_neox.layers[3].register_forward_hook(
        lambda m, i, o: cap.__setitem__(3, (o[0] if isinstance(o, tuple) else o).detach().float()[0]))
    ids = tok(["1" + VH + "2"], padding="max_length", max_length=350, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        model(input_ids=ids)
    sae = S.load("piggen_topk_r32_layer3")
    return cap[3], sae.W_enc.detach().float(), sae.b_enc.detach().float(), sae.b_dec.detach().float()


def dense_df(x, We, be, bd, latents, neuron=None):
    """Top-32 activations of the given latents (0 when not selected) per token; '1', '2' and padding dropped;
    Position = token index + 1."""
    v, i = torch.relu((x - bd) @ We.T + be).topk(32, dim=-1)
    v, i = v.numpy(), i.numpy()
    d = {f"Neuron_{lt}": np.where(i == lt, v, 0.0).sum(axis=1) for lt in latents}
    if neuron is not None:
        d[f"Hidden_{neuron}"] = x[:, neuron].numpy()
    d["AminoAcid"] = list(("1" + VH + "2").ljust(350, "*"))
    df = pd.DataFrame(d)
    df = df[~df["AminoAcid"].isin({'1', '2', '*', 'X'})]
    df["Position"] = df.index + 1
    return df


def create_activation_plot(df, mask, series, legend_title, save_path, ylim0, figsize=(24, 6), annotate_j=None,
                           title_xlabel=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mask = mask[:len(df)]
    boundary_idx = mask.find('!')
    plot_df = df[df['Position'] <= boundary_idx].copy()          # VH only
    fig, ax = plt.subplots(figsize=figsize)
    cdr_colors = {'A': '#ff9999', 'B': '#99ccff', 'C': '#99ff99', 'Z': '#ffcc99', 'Y': '#ff99ff', 'W': '#ffb366'}
    for _, r in plot_df.iterrows():
        pos = r['Position'] - 1
        if pos < len(mask) and mask[pos] in cdr_colors:
            ax.axvspan(r['Position'] - 0.5, r['Position'] + 0.5, alpha=0.2, color=cdr_colors[mask[pos]])
    for col, label, color, marker, size in series:
        max_val = plot_df[col].abs().max()
        plot_df[f"{col}_norm"] = plot_df[col] / max_val
        ax.scatter(plot_df["Position"], plot_df[f"{col}_norm"], s=size, color=color, marker=marker, label=label, zorder=3)
        for p, val in zip(plot_df["Position"], plot_df[f"{col}_norm"]):
            ax.plot([p, p], [0, val], color=color, linewidth=1.5, alpha=0.7, zorder=2)
    ax.set_xticks(plot_df["Position"].tolist())
    ax.set_xticklabels(plot_df["AminoAcid"].tolist())
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=14)
    if title_xlabel:
        ax.set_title("VH Region", fontsize=20)
    if title_xlabel:
        ax.set_xlabel("Sequence Position (Amino Acid)", fontsize=26, labelpad=10)
    ax.set_ylabel("Normalised Activation", fontsize=26, labelpad=10)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.3)
    leg1 = ax.legend(title=legend_title, loc='upper left', bbox_to_anchor=(1.0, 1), fontsize=18, title_fontsize=20)
    extra = [leg1]   # legends added with add_artist are outside the tight bbox unless listed
    names = {'A': 'CDRH1', 'B': 'CDRH2', 'C': 'CDRH3'}
    vis = {mask[r - 1] for r in plot_df['Position'] if r - 1 < len(mask)}
    handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=cdr_colors[k], markersize=15, label=v)
               for k, v in names.items() if k in vis]
    ax.legend(handles=handles, title="CDR Regions", loc='center left', bbox_to_anchor=(1.01, 0.4),
              fontsize=18, title_fontsize=20)
    ax.add_artist(leg1)
    if ylim0:
        ax.set_ylim(0, None)
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)
    if annotate_j is not None:
        # Supplementary Figure 5: box around the heavy J residues and an arrow label
        from matplotlib.patches import Rectangle
        fig.canvas.draw()
        lo, hi = annotate_j
        labs = [t for t, p in zip(ax.get_xticklabels(), plot_df['Position']) if lo <= p <= hi]
        bbs = [t.get_window_extent() for t in labs]
        x0, x1 = min(b.x0 for b in bbs), max(b.x1 for b in bbs)
        y0, y1 = min(b.y0 for b in bbs), max(b.y1 for b in bbs)
        inv = fig.transFigure.inverted()
        (fx0, fy0), (fx1, fy1) = inv.transform((x0 - 4, y0 - 4)), inv.transform((x1 + 4, y1 + 4))
        fig.patches.append(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, transform=fig.transFigure,
                                     fill=False, edgecolor='black', linewidth=1.5))
        extra.append(fig.text(fx1 + 0.004, (fy0 + fy1) / 2, "← Heavy J region", va="center", ha="left", fontsize=18))
    fig.savefig(save_path, bbox_inches='tight', dpi=300, bbox_extra_artists=extra)
    plt.close(fig)


def fit(im, w, h):
    """Pad with white (or crop) symmetrically to exactly w x h."""
    from PIL import Image
    im = im.convert("RGB")
    canvas = Image.new("RGB", (max(w, im.width), max(h, im.height)), "white")
    canvas.paste(im, ((canvas.width - im.width) // 2, (canvas.height - im.height) // 2))
    l, t = (canvas.width - w) // 2, (canvas.height - h) // 2
    return canvas.crop((l, t, l + w, t + h))


def render_to(df, mask, target_w, target_h, path, **kw):
    """Render with a tight bbox, adjusting figsize until the PNG is target_w x target_h px at 300 dpi (legends have
    fixed size, so the axes absorb the change); pad or crop the last few pixels."""
    from PIL import Image
    fw, fh = 24.0, 6.0
    for it in range(4):
        create_activation_plot(df, mask, save_path=path, figsize=(fw, fh), **kw)
        im = Image.open(path); im.load()
        if abs(im.width - target_w) <= 2 and abs(im.height - target_h) <= 2:
            break
        fw += (target_w - im.width) / 300.0
        fh += (target_h - im.height) / 300.0
    os.remove(path)
    return fit(im, target_w, target_h)


def main(argv=None):
    from PIL import Image
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    out = argv[0]; os.makedirs(out, exist_ok=True)
    mask = cdr_mask(VH, VL, **CDRS)
    x, We, be, bd = activations()
    df = dense_df(x, We, be, bd, [4280, 3054, 532, 3098], neuron=45)

    # Figure 1: 6610 x 1519 px, no title or x-axis label
    im = render_to(df, mask, 6610, 1519, os.path.join(out, "fig1_tmp.png"),
                   series=[("Neuron_4280", "Latent 4280", '#1f77b4', 'o', 100), ("Hidden_45", "Neuron 45", '#d62728', 'o', 50)],
                   legend_title="Activations", ylim0=False, title_xlabel=False)
    f1 = os.path.join(out, "combined_latent_neuron_plot_crop_preln.png")
    im.save(f1); print("Figure 1:", f1, im.size)

    # Supplementary Figure 5: 6610 x 1772 render with the J-region box, downscaled to 1600 x 429 (96 dpi)
    j_lo = 1 + VH.rfind('IWGQG') + 1        # Position of the I of the J motif
    im = render_to(df, mask, 6610, 1772, os.path.join(out, "s5_tmp.png"),
                   series=[("Neuron_3054", "Latent 3054", '#1f77b4', 'o', 100), ("Neuron_532", "Latent 532", '#d62728', 's', 100),
                           ("Neuron_3098", "Latent 3098", '#2ca02c', '^', 100)],
                   legend_title="Latents", ylim0=True, annotate_j=(j_lo, 10_000))
    im = fit(im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS), 1600, 429)
    f5 = os.path.join(out, "Your_paragraph_text_preln.png")
    im.save(f5, dpi=(96.012, 96.012)); print("Supplementary Figure 5:", f5, im.size)


if __name__ == "__main__":
    main()
