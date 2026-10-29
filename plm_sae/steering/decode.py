"""Sampling loop equivalent to Hugging Face generate(do_sample=True, temperature, top_p), with per-row stopping and
batch compaction. The logit processing follows transformers 4.57 (GenerationMixin._sample, TemperatureLogitsWarper,
TopPLogitsWarper; Apache License 2.0, Copyright 2020 The HuggingFace Inc. team, THIRD_PARTY_NOTICES.md): float32
logits / temperature, ascending sort, cumulative softmax, drop cumprob <= 1 - top_p keeping the top token, softmax,
one draw per row. Finished rows are pad-filled after their stop token. With compact=False and crn=None the output is
token-identical to Hugging Face generate with the same seed; with compact=True finished rows are dropped from the
batch and the KV cache, which changes which random numbers each row receives but not its sampling distribution.

crn (LongTensor [B]): common random numbers. Rows with the same crn index share one uniform draw per step and sample by
inverse CDF, so every configuration (latent, alpha) of a call sees the same random stream. bad_ids and forced_eos
reproduce NoBadWordsLogitsProcessor (single-token bad words) and ForcedEOSTokenLogitsProcessor (EOS at the last
position) for IgLM. The steerer's per-row vectors and start (S) and end (E) positions are compacted with the batch."""
import torch


def _top_p(scores, top_p):
    """TopPLogitsWarper of transformers 4.57 (Apache License 2.0)."""
    sl, si = torch.sort(scores, descending=False)
    cp = sl.softmax(dim=-1).cumsum(dim=-1)
    rm = cp <= (1 - top_p); rm[..., -1:] = 0
    return scores.masked_fill(rm.scatter(1, si, rm), -float("inf"))

def _select(past, idx):
    if hasattr(past, "batch_select_indices"): past.batch_select_indices(idx); return past
    return tuple(tuple(t.index_select(0, idx) for t in layer) for layer in past)

@torch.no_grad()
def sample(m, ctx, B, temp, top_p, max_new, stop_ids, pad, steerer=None, compact=True, compact_frac=0.2,
           crn=None, bad_ids=None, forced_eos=None):
    dev = ctx.device; L0 = ctx.shape[1]
    out = torch.full((B, L0 + max_new), pad, dtype=torch.long, device=dev); out[:, :L0] = ctx
    stop = torch.tensor(stop_ids, device=dev)
    rows = torch.arange(B, device=dev)              # original row index of each live batch slot
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    V = steerer.V if steerer is not None else None
    S = steerer.S if steerer is not None else None
    E = getattr(steerer, "E", None) if steerer is not None else None
    if crn is not None: crn_live = crn.to(dev); ncol = int(crn_live.max()) + 1
    inp, past = ctx.repeat(B, 1), None
    for t in range(max_new):
        if steerer is not None: steerer.p0 = 0 if t == 0 else L0 + t - 1
        o = m(input_ids=inp, past_key_values=past, use_cache=True)
        past = o.past_key_values
        sc = o.logits[:, -1, :].to(dtype=torch.float32)
        if bad_ids is not None: sc[:, bad_ids] = -float("inf")
        if forced_eos is not None and t == max_new - 1:
            sc = torch.full_like(sc, -float("inf")); sc[:, forced_eos] = 0
        if temp != 1.0: sc = sc / temp
        if top_p < 1.0: sc = _top_p(sc, top_p)
        pr = sc.softmax(dim=-1)
        if crn is None:
            nxt = torch.multinomial(pr, num_samples=1).squeeze(1)
        else:
            cdf = pr.cumsum(dim=-1)
            u = torch.rand(ncol, device=dev)[crn_live] * cdf[:, -1]
            nxt = (cdf < u[:, None]).sum(dim=-1).clamp_(max=pr.shape[1] - 1)
        nxt = torch.where(alive, nxt, torch.full_like(nxt, pad))
        out[rows, L0 + t] = nxt
        alive = alive & ~torch.isin(nxt, stop)
        n_alive = int(alive.sum())
        if n_alive == 0: break
        if compact and n_alive <= (1 - compact_frac) * alive.numel():
            keep = alive.nonzero().squeeze(1)
            past = _select(past, keep); rows = rows[keep]; nxt = nxt[keep]; alive = alive[keep]
            if crn is not None: crn_live = crn_live[keep]
            if steerer is not None:
                steerer.V = steerer.V[keep]
                if steerer.S is not None: steerer.S = steerer.S[keep]
                if getattr(steerer, "E", None) is not None: steerer.E = steerer.E[keep]
        inp = nxt[:, None]
    if steerer is not None:
        steerer.V = V; steerer.S = S; steerer.p0 = 0
        if E is not None: steerer.E = E
    return out[:, :L0 + t + 1]
