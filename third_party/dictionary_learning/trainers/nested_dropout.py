# From John J. Yang's reticular-sae code (private repository; reused with the author's permission), Apache License 2.0,
# Copyright 2025 John Yang (LICENSE-nested-dropout, LICENSE-APACHE-2.0). Modified for plm-sae: the orthonormality term
# is computed only when ortho_alpha != 0, which saves memory and leaves the loss unchanged. See NOTICE.md.
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from collections import namedtuple
from typing import Optional
from math import isclose

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


def apply_temperature(probabilities: list[float], temperature: float) -> list[float]:
    """
    Apply temperature scaling to a list of probabilities using PyTorch.

    Args:
        probabilities (list[float]): Initial probability distribution
        temperature (float): Temperature parameter (> 0)

    Returns:
        list[float]: Scaled and normalized probabilities
    """
    probs_tensor = t.tensor(probabilities, dtype=t.float32)
    logits = t.log(probs_tensor)
    scaled_logits = logits / temperature
    scaled_probs = t.nn.functional.softmax(scaled_logits, dim=0)

    return scaled_probs.tolist()


class NestedDropoutBatchTopKSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(t.zeros(dict_size))
        self.W_dec = nn.Parameter(t.nn.init.kaiming_uniform_(t.empty(dict_size, activation_dim)))
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.W_dec.data = set_decoder_norm_to_unit_norm(
            self.W_dec.data.T, activation_dim, dict_size
        ).T
        self.W_enc.data = self.W_dec.data.clone().T

    def encode(self, x: t.Tensor, return_active: bool = False, use_threshold: bool = True):
        post_relu_feat_acts_BF = nn.functional.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)
        else:
            # Flatten and perform batch top-k
            flattened_acts = post_relu_feat_acts_BF.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)

            encoded_acts_BF = (
                t.zeros_like(post_relu_feat_acts_BF.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts_BF.shape)
            )

        # max_act_index = self.group_indices[self.active_groups]
        # encoded_acts_BF[:, max_act_index:] = 0

        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor, slice_allowed: bool = False) -> t.Tensor:
        """Decodes the feature vector using the appropriate slice of W_dec."""
        num_features = x.shape[-1]
        if slice_allowed:
            return x @ self.W_dec[:num_features, :] + self.b_dec
        else:
            return x @ self.W_dec + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @t.no_grad()
    def scale_biases(self, scale: float):
        self.b_enc.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(cls, path, k=None, device=None, **kwargs) -> "NestedDropoutBatchTopKSAE":
        #state_dict = t.load(path, map_location=device) ###Changed the original code to remove the ae. prefix in the code
        ckpt = t.load(path, map_location=device, weights_only=False)
        raw = ckpt.get("state_dict", ckpt)

        # strip the "ae." prefix if present
        state_dict = {}
        for key, val in raw.items():
            if key.startswith("ae."):
                clean_key = key[len("ae."):]
            else:
                clean_key = key
            state_dict[clean_key] = val
            
        activation_dim, dict_size = state_dict["W_enc"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        # group_sizes = state_dict["group_sizes"].tolist()

        autoencoder = cls(activation_dim, dict_size, k=k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class NestedDropoutBatchTopKTrainer(SAETrainer):
    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        p_L_over_p_1: float,
        dict_class: type = NestedDropoutBatchTopKSAE,
        ortho_alpha: float = 0.0,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        wandb_name: str = "NestedDropoutBatchTopKSAE",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.ae = dict_class(activation_dim, dict_size, k)

        self.device = "cuda" if t.cuda.is_available() else "cpu" #NOTE: This is wrong during distributed training.

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5
        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long)
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        # Initialize orthonormality loss
        self.ortho_alpha = ortho_alpha

        # Initialize nested dropout parameters
        self.p_L_over_p_1 = p_L_over_p_1
        self.init_p_b(dict_size, p_L_over_p_1, self.device)

    def init_p_b(self, L, ratio, device='cpu'):
        """
        ratio = p(L)/p(1).  Solve rho^(L-1) = ratio -> rho = ratio^(1/(L-1)).
        Return normalized truncated geometric probabilities p_1,...,p_L.
        Note: We use the form of the geometric distribution p(b) = (1-rho)*rho^(b-1).
        """
        # Solve for rho
        rho = ratio ** (1.0/(L-1))

        # Build unnormalized
        unnorm = [(1-rho)*rho**(b-1) for b in range(1, L+1)]
        s = sum(unnorm)
        dist = [u/s for u in unnorm]
        print(f'p(1) = {dist[0]:.6e}, p(L) = {dist[-1]:.6e}')
        self.p_b = t.tensor(dist, dtype=t.float32, device=device)

        # Precompute cumulative distribution
        self.cumsum_p_b = t.cat([
            t.tensor([0.0], device=device),
            self.p_b.cumsum(dim=0)
        ], dim=0)

    def orthonormality_loss(self, W: t.Tensor) -> t.Tensor:
        row_norms = W.norm(dim=1, keepdim=True)
        row_norms = row_norms.clamp(min=1e-6) 
        W_norm = W / row_norms

        cosine_sim = W_norm @ W_norm.T
        identity = t.eye(W.shape[0], device=W.device)
        loss = ((cosine_sim - identity) ** 2).sum()
        return loss / (W.shape[0] * (W.shape[0] - 1))


    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        # Ensure the dead-feature mask lives on the same device as the activations
        dead_features = self.num_tokens_since_fired.to(post_relu_acts_BF.device) >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # We don't want to apply the bias
            x_reconstruct_aux = auxk_acts_BF @ self.ae.W_dec
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, f: t.Tensor):
        device_type = "cuda" if f.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = f[f > 0]

            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=t.float32)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def return_boundaries(self, x: t.Tensor, mask: t.Tensor):
        """
        Given an (B, F) tensor `x` and a corresponding boolean mask of the same shape,
        return a boundaries tensor per sample as well as a per–sample count of valid boundaries.
        
        The returned boundaries tensor has shape (B, max_valid + 2), where each row is
        [0, sorted_valid_indices, F]. Here F (the feature dimension) is used as a sentinel.
        boundaries_lengths is a (B,) tensor that counts the number of valid (i.e. active)
        indices in each sample.
        """
        B, F = x.shape
        # Create a column index for features, shape (B, F)
        cols = t.arange(F, device=x.device).unsqueeze(0).expand(B, F)
        # Where mask is False, assign sentinel value F; valid entries retain the column indices.
        active_cols = t.where(mask, cols, t.full_like(cols, F))
        # Sort each row so that valid indices (< F) come first.
        sorted_active, _ = t.sort(active_cols, dim=1)
        valid_mask = sorted_active < F
        # Count valid indices for each sample.
        boundaries_lengths = valid_mask.sum(dim=1)
        max_valid = int(boundaries_lengths.max().item())
        # For a compact representation, take only the first max_valid valid entries.
        valid_boundaries = sorted_active[:, :max_valid]
        # Prepend 0 and append F to every row.
        start_boundary = t.zeros((B, 1), device=x.device, dtype=t.long)
        end_boundary = t.full((B, 1), F, device=x.device, dtype=t.long)
        boundaries_tensor = t.cat([start_boundary, valid_boundaries, end_boundary], dim=1)
        return boundaries_tensor, boundaries_lengths

    def loss(self, x, step=None, logging=False):
        # Encode (same as before)
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=False
        )
        B, L = f.shape
        target_device = x.device
        
        if target_device != self.device:
            self.device = target_device

        # Keep bookkeeping tensors on the active device to avoid CPU/GPU mismatches
        self.num_tokens_since_fired = self.num_tokens_since_fired.to(target_device)
        
        # Threshold update (same as before)
        if step is not None and step > self.threshold_start_step:
            self.update_threshold(f)
        
        # Ensure p_b setup (same as before)
        if len(self.p_b) != L:
            print(f"Warning: p_b length ({len(self.p_b)}) != L ({L}). Re-initializing.")
            self.init_p_b(L, self.p_L_over_p_1, target_device)
        elif self.p_b.device != target_device:
            self.init_p_b(L, self.p_L_over_p_1, target_device)
        
        # Generate non-zero mask for each sample
        non_zero_mask = f != 0
        
        # Get boundaries and boundary lengths for each sample
        boundaries_tensor, boundaries_lengths = self.return_boundaries(f, non_zero_mask)
        
        # Compute the initial reconstructions (just the bias for all samples)
        x_hat_prefix = self.ae.b_dec.unsqueeze(0).expand(B, -1)
        
        # Initialize loss accumulator for each sample
        sample_losses = t.zeros(B, device=target_device)
        
        # Maximum number of boundaries across all samples
        max_boundaries = boundaries_tensor.size(1)
        
        # Process each boundary pair across all samples simultaneously
        for j in range(max_boundaries - 1):
            # Extract start and end levels for this boundary pair
            start_levels = boundaries_tensor[:, j]
            end_levels = boundaries_tensor[:, j+1]
            
            # Initialize delta_x_hat for all samples
            delta_x_hat = t.zeros_like(x_hat_prefix)
            
            # For each sample, if end_level < L, get the contribution from the feature at end_level-1
            valid_ends = end_levels < L
            if valid_ends.any():
                # For samples with valid ends, get indices and values
                valid_indices = t.nonzero(valid_ends, as_tuple=True)[0]
                feature_indices = end_levels[valid_indices]
                
                # Get feature values directly using advanced indexing instead of gather
                feature_values = f[valid_indices, feature_indices]
                
                # Get corresponding decoder weights
                decoder_weights = self.ae.W_dec[feature_indices]
                
                # Compute contributions
                contributions = feature_values.unsqueeze(1) * decoder_weights
                
                # Add contributions to delta_x_hat
                delta_x_hat[valid_indices] = contributions
            
            # Update reconstructions
            x_hat = x_hat_prefix + delta_x_hat
            
            # Compute MSE for this boundary
            mse = ((x - x_hat) ** 2).mean(dim=1)
            
            # Compute probability mass for this boundary range
            block_probs = self.cumsum_p_b[end_levels] - self.cumsum_p_b[start_levels]
            
            # Add weighted MSE to sample losses, but only for samples where j < boundaries_length
            valid_samples = t.arange(B, device=target_device) < boundaries_lengths
            sample_losses[valid_samples] += block_probs[valid_samples] * mse[valid_samples]
            
            # Update x_hat_prefix for next iteration
            x_hat_prefix = x_hat
        
        # Average loss over batch
        nested_loss_estimate = sample_losses.mean()
        
        # Rest of the function remains the same (auxiliary loss calculation, etc.)
        x_reconstruct_full = self.ae.decode(f)
        full_diff = (x - x_reconstruct_full)
        full_mse_loss = ((full_diff) ** 2).mean()
        auxk_loss = self.get_auxiliary_loss(full_diff.detach(), post_relu_acts_BF)
        
        # Combine losses
        if isinstance(auxk_loss, t.Tensor):
            loss = nested_loss_estimate + self.auxk_alpha * auxk_loss
        else:
            loss = nested_loss_estimate + self.auxk_alpha * auxk_loss if auxk_loss != -1 else nested_loss_estimate
        
        # Bookkeeping and logging (same as before)
        self.effective_l0 = self.k
        num_tokens_in_step = x.size(0)
        # Ensure tensors used for bookkeeping are on the correct device if needed
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool) # Inherits device from num_tokens_since_fired
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        if t.distributed.is_initialized():
            # Check if tensor is on CPU but training is on GPU
            if self.num_tokens_since_fired.device.type == 'cpu' and t.cuda.is_available():
                # Move to same device as model/input tensors
                device_tensor = self.num_tokens_since_fired.to('cuda')
                t.distributed.all_reduce(device_tensor, op=t.distributed.ReduceOp.MIN)
                self.num_tokens_since_fired = device_tensor.cpu()  # Move back to CPU
            else:
                # Tensor already on correct device
                t.distributed.all_reduce(self.num_tokens_since_fired, op=t.distributed.ReduceOp.MIN)

        # But, final final loss adds the orthonormality loss!
        # The orthonormality term is only ever added when ortho_alpha != 0. It
        # materialises a d_sae x d_sae matrix (~151 MB at the shipped dict_size=6144,
        # ~2.42 GB at the r=32 dict_size=24576 for p-IgGen), so skip the computation
        # entirely when it cannot contribute to the loss to avoid OOM. All shipped
        # checkpoints trained with ortho_alpha=0.
        if self.ortho_alpha != 0:
            orth_loss = self.orthonormality_loss(self.ae.W_dec)
            loss = loss + self.ortho_alpha * orth_loss
        else:
            orth_loss = t.tensor(0.0, dtype=loss.dtype, device=loss.device)

        if not logging:
            return loss
        else:
            from collections import namedtuple
            LossLog = namedtuple("LossLog", ["x", "x_hat_full", "f", "losses"])
            return LossLog(
                x,
                x_reconstruct_full,
                f,
                {
                    "l2_loss": full_mse_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                    "orth_loss": orth_loss.item(),
                },
            )


    def update(self, step, x):
        if step == 0:
            median = self.geometric_median(x)
            if t.distributed.is_initialized():
                # Only synchronize the initial median if needed
                t.distributed.broadcast(median, src=0)
            self.ae.b_dec.data = median

        target_device = x.device
        if target_device != self.device:
            self.device = target_device

        # x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T, self.ae.W_dec.grad.T, self.ae.activation_dim, self.ae.dict_size
        ).T
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "NestedDropoutBatchTopKTrainer",
            "dict_class": "NestedDropoutBatchTopKSAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "p_L_over_p_1": self.p_L_over_p_1,
            "ortho_alpha": self.ortho_alpha,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = t.zeros_like(guess)
        weights = t.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break

        return guess
