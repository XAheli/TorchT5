"""
Training utilities for T5 model
Includes learning rate scheduling, checkpointing, and optimization helpers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
import json
import math
from typing import Dict, Any, Optional, Union


def get_linear_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    loss: float,
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None
):
    """Save training checkpoint"""
    os.makedirs(checkpoint_path, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': loss
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    # Save the checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_path, 'pytorch_model.bin'))
    
    # Save model configuration
    if hasattr(model, 'config'):
        config_dict = model.config.to_dict() if hasattr(model.config, 'to_dict') else model.config.__dict__
        with open(os.path.join(checkpoint_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> Optional[Dict[str, Any]]:
    """Load training checkpoint"""
    checkpoint_file = os.path.join(checkpoint_path, 'pytorch_model.bin')
    
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint not found at {checkpoint_file}")
        return None
    
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def freeze_encoder(model: nn.Module):
    """Freeze encoder parameters for fine-tuning decoder only"""
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True


def set_dropout(model: nn.Module, dropout_rate: float):
    """Set dropout rate for all dropout layers"""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate


def gradient_checkpointing_enable(model: nn.Module):
    """Enable gradient checkpointing to save memory"""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Compute cross-entropy loss with label smoothing option"""
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss


def label_smoothed_nll_loss(
    lprobs: torch.Tensor, 
    target: torch.Tensor, 
    epsilon: float = 0.1, 
    ignore_index: int = -100
) -> torch.Tensor:
    """Compute label-smoothed negative log-likelihood loss"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    
    if ignore_index >= 0:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    
    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)
    
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    
    return loss.mean()


class WarmupLR:
    """Learning rate warmup scheduler"""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, initial_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.initial_lr + (self.base_lrs[i] - self.initial_lr) * (self.step_count / self.warmup_steps)
                param_group['lr'] = lr


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    optimizer_type: str = "adamw"
) -> optim.Optimizer:
    """Create optimizer with proper weight decay handling"""
    # Parameters that should not have weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon
        )
    elif optimizer_type.lower() == "adam":
        return optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_epsilon
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")