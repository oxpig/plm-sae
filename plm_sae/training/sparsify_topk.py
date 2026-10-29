"""Train the expansion-32 TopK SAEs of Section 4.1 with the EleutherAI sparsify library (k = 32): one SAE on each
p-IgGen block output (gpt_neox.layers.0-3; layer 3 is the SAE of Section 4.1). Settings are the library defaults plus
SaeConfig(expansion_factor=32), batch_size=64 and padding to 350 tokens; the model runs in bfloat16.
The run directory gets gpt_neox.layers.<L>/sae.safetensors + cfg.json (sparsify's format) and
piggen_topk_r32_layer<L>.safetensors + .json (the format of plm_sae.sae; copy them into <data_dir>/checkpoints/ to
use them by name).

Run: python -m plm_sae.training.sparsify_topk [--data_path CSV] [--run_name DIR]   (scripts/training/train_sparsify.sh)"""
import argparse
import datetime
import json
import os
import shutil

import torch
from datasets import load_dataset
from sparsify import SaeConfig, SaeTrainer, TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from plm_sae import models, paths as P


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_path", default=str(P.dataset("oas_sae_training_dataset_unwrapped.csv")),
                    help="CSV with a `text` column of paired sequences (VH+VL) without the 1 and 2 tokens")
    ap.add_argument("--run_name", default=str(P.work("training", "sparsify_topk_r32", f"{datetime.datetime.now()}")))
    a = ap.parse_args()
    os.environ["WANDB_DISABLED"] = "true"
    tok = AutoTokenizer.from_pretrained(models.PIGGEN, revision=models.PIGGEN_REVISION)
    data = load_dataset("csv", data_files=a.data_path, split="train")
    data = data.map(lambda x: tok([f"1{t}2" for t in x["text"]], padding="max_length", max_length=350),
                    batched=True, num_proc=12).with_format("torch", columns=["input_ids"])
    gpt = AutoModelForCausalLM.from_pretrained(models.PIGGEN, revision=models.PIGGEN_REVISION,
                                               device_map={"": "cuda"}, torch_dtype=torch.bfloat16)
    cfg = TrainConfig(SaeConfig(expansion_factor=32, num_latents=0), batch_size=64, run_name=a.run_name)
    print(cfg)
    trainer = SaeTrainer(cfg, data, gpt)
    trainer.fit()
    trainer.save()
    for layer in range(4):
        d = os.path.join(a.run_name, f"gpt_neox.layers.{layer}")
        name = os.path.join(a.run_name, f"piggen_topk_r32_layer{layer}")
        sc = json.load(open(os.path.join(d, "cfg.json")))
        shutil.copyfile(os.path.join(d, "sae.safetensors"), name + ".safetensors")
        json.dump(dict(name=os.path.basename(name), model="p-IgGen", architecture="TopK (EleutherAI sparsify)",
                       layer=layer, site=f"output of p-IgGen layer {layer} (gpt_neox.layers.{layer}), raw",
                       dictionary_size=sc["num_latents"] or 32 * sc["d_in"], activation_dim=sc["d_in"],
                       expansion_factor=sc["expansion_factor"], k=sc["k"], sparsify_cfg=sc),
                  open(name + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
