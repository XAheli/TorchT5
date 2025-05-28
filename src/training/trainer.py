"""
T5 Training Implementation
Comprehensive trainer with optimization, scheduling, and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import os
import json
import time
from tqdm import tqdm
import logging
from dataclasses import dataclass

from ..model.t5_model import T5ForConditionalGeneration
from ..model.config import T5Config
from .utils import get_linear_schedule_with_warmup, save_checkpoint, load_checkpoint


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./outputs"
    resume_from_checkpoint: Optional[str] = None
    use_wandb: bool = False
    eval_during_training: bool = True


class T5Trainer:
    """
    Comprehensive T5 trainer with optimization and evaluation
    """
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup wandb if requested
        if self.config.use_wandb:
            self._setup_wandb()
            
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup AdamW optimizer with weight decay"""
        # Separate parameters that should and shouldn't have weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            import wandb
            wandb.init(
                project="t5-from-scratch",
                config=self.config.__dict__,
                dir=self.config.output_dir
            )
        except ImportError:
            self.logger.warning("wandb not installed. Install with: pip install wandb")
            self.config.use_wandb = False

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Number of epochs: {self.config.num_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )
            
            for step, batch in enumerate(progress_bar):
                loss = self.training_step(batch)
                epoch_loss += loss
                total_loss += loss
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    self.logger.info(f"Step {self.global_step}: loss = {avg_loss:.4f}")
                    
                    if self.config.use_wandb:
                        import wandb
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": self.global_step
                        })
                    
                    total_loss = 0
                
                # Evaluation
                if (self.config.eval_during_training and 
                    self.eval_dataloader is not None and 
                    self.global_step % self.config.eval_steps == 0):
                    self.evaluate()
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            self.logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # End of epoch evaluation
            if self.config.eval_during_training and self.eval_dataloader is not None:
                eval_loss = self.evaluate()
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint(is_best=True)
        
        self.logger.info("Training completed!")
        
        # Final save
        self.save_checkpoint(final=True)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs['loss']
        
        # Backward pass
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
        
        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def evaluate(self) -> float:
        """Evaluate model on validation set"""
        if self.eval_dataloader is None:
            return float('inf')
        
        self.logger.info("Running evaluation...")
        self.model.eval()
        
        total_eval_loss = 0
        num_eval_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                total_eval_loss += loss.item()
                num_eval_steps += 1
        
        avg_eval_loss = total_eval_loss / num_eval_steps
        self.logger.info(f"Evaluation loss: {avg_eval_loss:.4f}")
        
        if self.config.use_wandb:
            import wandb
            wandb.log({
                "eval_loss": avg_eval_loss,
                "step": self.global_step
            })
        
        self.model.train()
        return avg_eval_loss

    def save_checkpoint(self, is_best: bool = False, final: bool = False):
        """Save model checkpoint"""
        if final:
            checkpoint_path = os.path.join(self.config.output_dir, "final_model")
        elif is_best:
            checkpoint_path = os.path.join(self.config.output_dir, "best_model")
        else:
            checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            global_step=self.global_step,
            loss=self.best_eval_loss,
            checkpoint_path=checkpoint_path
        )
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint_data = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )
        
        if checkpoint_data:
            self.epoch = checkpoint_data['epoch']
            self.global_step = checkpoint_data['global_step']
            self.best_eval_loss = checkpoint_data.get('loss', float('inf'))
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
            self.logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")


def train_model(
    model: T5ForConditionalGeneration,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None
) -> T5Trainer:
    """Convenience function to train T5 model"""
    if config is None:
        config = TrainingConfig()
    
    trainer = T5Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )
    
    trainer.train()
    return trainer