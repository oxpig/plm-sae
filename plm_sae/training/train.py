# Modified from training/train.py of reticular-sae by John J. Yang (github.com/johnyang101/reticular-sae), Apache License 2.0.
# Changes for this project: plm_sae/training/NOTICE.md.
"""Train an SAE on base-model activations (Hydra configuration in configs/training/, PyTorch Lightning).

Run (scripts/training/train_sae.sh gives every override of the paper's runs):
  python -m plm_sae.training.train datamodule=piggen sae_trainer_cfg=batch_top_k expansion_factor=8 ... save_dir=DIR
Writes DIR/sae-step-<step>.ckpt every save_steps steps and, at the end, DIR/<name>.safetensors + DIR/<name>.json
(the checkpoint format of the data record, read by plm_sae.sae)."""
from typing import List, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.trainers.nested_dropout import NestedDropoutBatchTopKSAE, NestedDropoutBatchTopKTrainer
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from plm_sae import paths as P
from plm_sae.training.datamodule import LitData_PLM_Datamodule as PLMDatamodule
from plm_sae.training.trainer import CustomModelCheckpointCallback_DL, SAETrainerLM, export_sae
from plm_sae.training.utils import calculate_training_params, update_config_with_training_params, validate_dict_size

# configs/training/*.yaml refer to locations as ${plm_sae.activations:training,piggen_layer3_100k} and
# ${plm_sae.work:training,NAME}
OmegaConf.register_new_resolver("plm_sae.activations", lambda *p: str(P.activations(*p)), replace=True)
OmegaConf.register_new_resolver("plm_sae.work", lambda *p: str(P.work(*p)), replace=True)
CONFIG_DIR = str(P.REPO / "configs" / "training")

# sae_trainer_cfg.trainer_class -> (trainer, dictionary class)
TRAINERS = {
    "TopKTrainer": (TopKTrainer, AutoEncoderTopK),                                 # TopK
    "BatchTopKTrainer": (BatchTopKTrainer, BatchTopKSAE),                          # Batch TopK
    "OrderedTrainer": (NestedDropoutBatchTopKTrainer, NestedDropoutBatchTopKSAE),  # Ordered (nested dropout)
}
MODEL = {"piggen": "p-IgGen", "iglm": "IgLM", "progen2_oas": "ProGen2-OAS"}   # model names of the checkpoint json


def setup_datamodule(cfg: DictConfig, training_params) -> Tuple[PLMDatamodule, DictConfig]:
    datamodule = PLMDatamodule(
        optimized_dir=cfg.datamodule.optimized_dir,
        batch_size=cfg.computed.datamodule_batch_size,
        model=cfg.datamodule.model_type,
        layer=cfg.datamodule.layer_number,
        activation_dim=cfg.sae_trainer_cfg.activation_dim,
        normalize_activations=cfg.datamodule.normalize_activations,
        norm_factor_avging_steps=cfg.datamodule.norm_factor_avging_steps,
        center_activations=cfg.datamodule.center_activations,
        shuffle=cfg.datamodule.shuffle,
        seed=0 if cfg.fix_batch_order else cfg.seed,
        norm_factor=cfg.datamodule.norm_factor,
    )
    datamodule.setup()
    if cfg.datamodule.normalize_activations:
        training_params.norm_factor = datamodule.norm_factor
        cfg = update_config_with_training_params(cfg, training_params)
        print(f"Normalization factor: {cfg.computed.norm_factor:.4f}")
    return datamodule, cfg


def trainer_cfg(cfg: DictConfig) -> dict:
    """sae_trainer_cfg with the trainer and dictionary classes, the schedule length and the seed."""
    d = dict(cfg.sae_trainer_cfg)
    if d["trainer_class"] not in TRAINERS:
        raise ValueError(f"unknown trainer_class {d['trainer_class']}; one of {list(TRAINERS)}")
    d["trainer_class"], d["dict_class"] = TRAINERS[d["trainer_class"]]
    d.update(steps=cfg.computed.total_steps, seed=cfg.seed)
    return d


def create_trainer(cfg: DictConfig, callbacks: List[pl.Callback]) -> pl.Trainer:
    common = dict(callbacks=callbacks, logger=False, log_every_n_steps=cfg.log_steps, enable_progress_bar=True,
                  enable_checkpointing=True, fast_dev_run=cfg.fast_dev_run)
    if "steps" in cfg and cfg.steps:
        # SAETrainerLM steps its optimiser itself, so trainer.global_step never advances and max_steps alone would not
        # stop the run; limit_train_batches does.
        return pl.Trainer(max_steps=cfg.steps, limit_train_batches=cfg.steps, **common)
    if cfg.epochs:
        return pl.Trainer(max_epochs=cfg.epochs, **common)
    raise ValueError("either epochs or steps must be set")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    cfg.sae_trainer_cfg.dict_size = int(cfg.sae_trainer_cfg.activation_dim * cfg.expansion_factor)
    validate_dict_size(cfg.sae_trainer_cfg.activation_dim, cfg.expansion_factor, cfg.sae_trainer_cfg.dict_size)
    params = calculate_training_params(cfg)
    cfg = update_config_with_training_params(cfg, params)
    datamodule, cfg = setup_datamodule(cfg, params)
    print(OmegaConf.to_yaml(cfg))
    model = SAETrainerLM(sae_trainer_cfg=trainer_cfg(cfg))
    callbacks = [CustomModelCheckpointCallback_DL(dirpath=cfg.save_dir, filename="sae-step-{step:02d}.ckpt",
                                                  every_n_train_steps=cfg.save_steps)]
    create_trainer(cfg, callbacks).fit(model, datamodule=datamodule)
    s = cfg.sae_trainer_cfg
    meta = dict(model=MODEL[cfg.datamodule.model_type], layer=s.layer - 1, activation_dim=s.activation_dim,
                dictionary_size=s.dict_size, expansion_factor=cfg.expansion_factor, k=s.k, learning_rate=s.lr,
                warmup_steps=s.warmup_steps, auxk_alpha=s.auxk_alpha, seed=cfg.seed,
                norm_factor=cfg.computed.get("norm_factor"),
                site="final-layer-norm output (hidden_states[-1]) divided by norm_factor")
    print("exported", export_sae(model, cfg.save_dir, cfg.name, meta))


if __name__ == "__main__":
    main()
