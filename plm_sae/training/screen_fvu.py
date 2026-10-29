"""FVU, L0 and dead fraction of an SAE on the first records of its training activations: the ranking of the
learning-rate screens (Methods). The activations are those the SAE was trained on (training activation cache divided
by the model's norm factor, no centring), encoded in one call over all tokens of --n_seqs records.

Run: python -m plm_sae.training.screen_fvu SAE MODEL [--n_seqs 128] [--data DIR]
     SAE: .safetensors file (or record name); MODEL: piggen | IgLM | progen2_oas
(scripts/training/lr_screen.sh rank runs it over the screen SAEs.)"""
import argparse

import torch
from litdata import StreamingDataset

from plm_sae import models, paths as P, sae as S

CACHE = {"piggen": "piggen_layer3_100k", "IgLM": "iglm_layer3", "progen2_oas": "progen2oas_layer26"}


@torch.no_grad()
def evaluate(sae, data_dir, norm_factor, n_seqs, device):
    ds = StreamingDataset(str(data_dir), shuffle=False, drop_last=True)
    x = torch.stack([ds[i] for i in range(n_seqs)]).to(device)
    x = x.reshape(-1, x.shape[-1]) / norm_factor
    f = sae.encode(x)
    x_hat = sae.decode(f)
    mse = (x - x_hat).pow(2).sum(-1).mean().item()
    var = (x - x.mean(0)).pow(2).sum(-1).mean().item()
    return {"fvu": mse / var, "mse": mse, "var": var, "l0": (f != 0).float().sum(-1).mean().item(),
            "dead_frac": (f.sum(0) == 0).float().mean().item(), "n_tokens": x.shape[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sae")
    ap.add_argument("model", choices=models.MODELS)
    ap.add_argument("--n_seqs", type=int, default=128)
    ap.add_argument("--data", help="activation cache (default: the model's training cache)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sae = S.load(a.sae).to(dev)
    m = evaluate(sae, a.data or P.activations("training", CACHE[a.model]), models.NORM_FACTOR[a.model], a.n_seqs, dev)
    print(f"FVU={m['fvu']:.6f}  MSE={m['mse']:.6f}  Var={m['var']:.6f}  "
          f"L0={m['l0']:.3f}  dead={100 * m['dead_frac']:.1f}%  tokens={m['n_tokens']}")


if __name__ == "__main__":
    main()
