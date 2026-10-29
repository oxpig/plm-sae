# Modified from training/trainer.py of reticular-sae by John J. Yang (github.com/johnyang101/reticular-sae), Apache License 2.0.
# Changes for this project: plm_sae/training/NOTICE.md.
"""
PyTorch Lightning modules and callbacks for training sparse autoencoders.

This module provides Lightning modules for training various types of 
sparse autoencoders on protein language model (PLM) activations, as well as
a callback for model checkpointing and the export of the trained SAE as <name>.safetensors + <name>.json.

From reticular-sae, modified for this project (Apache License 2.0, see plm_sae/training/NOTICE.md).
"""

import json
import os
import torch
from typing import Dict, Any, Optional, List, Tuple

import pytorch_lightning as pl

from safetensors.torch import save_file

class SAETrainerLM(pl.LightningModule):
    """
    Lightning module for Sparse Autoencoder training.
    
    This class wraps SAE trainers from the dictionary_learning package
    to enable training with PyTorch Lightning.
    
    Args:
        sae_trainer_cfg: Configuration dictionary for the SAE trainer
        logging_steps: Frequency of logging training statistics (in steps)
    """

    def __init__(
        self,
        sae_trainer_cfg: Dict[str, Any],
        logging_steps: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # We handle optimization manually
        self.logging_steps = logging_steps
        self.sae_trainer_cfg = sae_trainer_cfg

        # Initialize the SAE trainer
        sae_trainer_class = self.sae_trainer_cfg.get("trainer_class")
        if not sae_trainer_class:
            raise ValueError("Trainer class not provided in sae_trainer_cfg")
            
        trainer_cfg = dict(self.sae_trainer_cfg)
        del trainer_cfg["trainer_class"]
        self.sae_trainer = sae_trainer_class(**trainer_cfg)
        self.ae = self.sae_trainer.ae

        # Counter for tracking steps (used for logging and scheduling)
        self.lm_step_counter = 0 

    def setup(self, stage):
        current_device = self.device  # Lightning's device context for this module/process
        self.sae_trainer.device = current_device
        self.ae.to(current_device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, d_model]
            
        Returns:
            Reconstructed tensor of same shape
        """
        return self.sae_trainer.ae(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Perform a single training step.
        
        Args:
            batch: Input batch of activations
            batch_idx: Index of the current batch
        """
        self.sae_trainer.update(self.lm_step_counter, batch)
        self.lm_step_counter += 1
    
    def on_train_batch_end(self, outputs: Optional[Any], batch: torch.Tensor, batch_idx: int) -> None:
        """
        Log statistics at the end of each training batch.
        
        Args:
            outputs: Outputs from the training step (unused)
            batch: The input batch
            batch_idx: Index of the current batch
        """
        super().on_train_batch_end(outputs, batch, batch_idx)
        # Belt-and-braces: trainer.global_step never advances under this module's
        # manual optimization (see create_trainer in plm_sae/training/train.py), so
        # max_steps alone cannot terminate the run. limit_train_batches is the
        # primary mechanism; this guard makes the intent explicit and is a no-op
        # once limit_train_batches is set correctly.
        if self.trainer.max_steps and self.trainer.max_steps > 0 and self.lm_step_counter >= self.trainer.max_steps:
            self.trainer.should_stop = True
        if self.lm_step_counter % self.logging_steps == 0:
            log = self.log_stats(batch)
            for key, value in log.items():
                self.log(f'train/{key}', value, on_step=True, on_epoch=False)
        

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Tuple of (optimizers, schedulers)
        """
        return [self.sae_trainer.optimizer], [self.sae_trainer.scheduler]
    
    def log_stats(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Compute and return training statistics.
        
        Args:
            batch: Current batch of data
            
        Returns:
            Dictionary of statistics to log
        """
        log = {}
        act, act_hat, f, losslog = self.sae_trainer.loss(batch, step=self.lm_step_counter, logging=True)

        # Calculate L0 sparsity (average number of non-zero features)
        l0 = (f != 0).float().sum(dim=-1).mean().item()
        
        # Calculate fraction of variance explained
        total_variance = torch.var(act, dim=0).sum()
        residual_variance = torch.var(act - act_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance
        log["frac_variance_explained"] = frac_variance_explained.item()
        
        # Log loss components
        log.update({
            f"{k}": v.cpu().item() if isinstance(v, torch.Tensor) else v 
            for k, v in losslog.items()
        })
        log["l0"] = l0
        
        # Get additional logging parameters from trainer
        trainer_log = self.sae_trainer.get_logging_parameters()
        for name, value in trainer_log.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            log[f"{name}"] = value

        return log
    
class CustomModelCheckpointCallback_DL(pl.Callback):
    """
    Custom model checkpointing callback that uses pl_module's lm_step_counter.
    
    This callback saves model checkpoints at specified intervals, using the
    model's internal step counter instead of the trainer's global step.
    
    Args:
        dirpath: Directory path to save the checkpoints
        filename: Filename format for the checkpoint (e.g., 'model-{step}.ckpt')
        every_n_train_steps: Save a checkpoint every n training steps
    """
    
    def __init__(self, dirpath: str, filename: str, every_n_train_steps: int):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.every_n_train_steps = every_n_train_steps

        # Ensure the save directory exists
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """
        Save checkpoint at the end of each batch if conditions are met.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            outputs: Outputs from the training step
            batch: The input batch
            batch_idx: Index of the current batch
        """
        # Use the model's internal step counter
        step = pl_module.lm_step_counter

        # Save checkpoint if it's time
        if step > 0 and step % self.every_n_train_steps == 0:
            checkpoint_path = os.path.join(self.dirpath, self.filename.format(step=step))
            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved at step {step}: {checkpoint_path}")


def export_sae(pl_module: pl.LightningModule, save_dir: str, name: str, meta: Dict[str, Any]) -> str:
    """Write the trained SAE as <save_dir>/<name>.safetensors (state_dict of the dictionary class) and <name>.json
    (meta plus dict_class, trainer_class and training_step), the format plm_sae.sae reads."""
    ae = pl_module.ae
    sd = {k: v.detach().cpu().contiguous() for k, v in ae.state_dict().items()}
    stem = os.path.join(save_dir, name)
    save_file(sd, stem + ".safetensors")
    cfg = pl_module.sae_trainer_cfg
    info = dict(meta, name=name, training_step=pl_module.lm_step_counter,
                dict_class=f"{type(ae).__module__}.{type(ae).__name__}",
                trainer_class=f"{cfg['trainer_class'].__module__}.{cfg['trainer_class'].__name__}",
                tensors={k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")} for k, v in sd.items()})
    with open(stem + ".json", "w") as f:
        json.dump(info, f, indent=1)
    return stem + ".safetensors"
