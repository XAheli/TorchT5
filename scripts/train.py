"""
Training script for T5 model from scratch
Supports both pre-training and fine-tuning
"""

import argparse
import os
import json
import torch
import yaml
from torch.utils.data import DataLoader

from src.model.config import T5Config
from src.model.t5_model import T5ForConditionalGeneration
from src.data.tokenizer import T5Tokenizer, download_t5_tokenizer
from src.data.dataset import T5TextDataset, T5PretrainingDataset, create_dataloader, load_dataset_from_json
from src.training.trainer import TrainingConfig, T5Trainer
from src.training.utils import count_parameters, get_model_size_mb


def parse_args():
    parser = argparse.ArgumentParser(description="Train T5 model from scratch")
    
    # Model and data arguments
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'])
    parser.add_argument('--tokenizer_model', type=str, help='Path to SentencePiece model file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data JSON')
    parser.add_argument('--eval_data', type=str, help='Path to evaluation data JSON')
    parser.add_argument('--task_type', type=str, default='general', 
                       choices=['general', 'translation', 'summarization', 'qa', 'pretraining'])
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    
    # Logging and checkpointing
    parser.add_argument('--logging_steps', type=int, default=100, help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint frequency')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation frequency')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Resume training from checkpoint')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    # Data processing
    parser.add_argument('--max_source_length', type=int, default=512, help='Max source sequence length')
    parser.add_argument('--max_target_length', type=int, default=50, help='Max target sequence length')
    
    return parser.parse_args()


def load_config(config_path: str, model_size: str) -> T5Config:
    """Load model configuration from YAML file"""
    if config_path and os.path.exists(config_path):
        return T5Config.from_yaml(config_path, model_size)
    else:
        # Use default configuration
        size_configs = {
            "small": {"d_model": 512, "d_ff": 2048, "num_layers": 6, "num_heads": 8},
            "base": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
            "large": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        }
        return T5Config(**size_configs[model_size])


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load tokenizer
    if args.tokenizer_model:
        if not os.path.exists(args.tokenizer_model):
            print(f"Tokenizer model not found at {args.tokenizer_model}")
            print("Downloading T5 tokenizer...")
            args.tokenizer_model = download_t5_tokenizer(f"t5-{args.model_size}")
    else:
        print("Downloading T5 tokenizer...")
        args.tokenizer_model = download_t5_tokenizer(f"t5-{args.model_size}")
    
    tokenizer = T5Tokenizer(args.tokenizer_model)
    print(f"Loaded tokenizer with vocab size: {tokenizer.get_vocab_size()}")
    
    # Load model configuration
    config = load_config(args.config, args.model_size)
    config.vocab_size = tokenizer.get_vocab_size()
    
    # Create model
    model = T5ForConditionalGeneration(config)
    
    # Print model info
    param_info = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    print(f"Model created with {param_info['total_parameters']:,} parameters ({model_size_mb:.2f} MB)")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    
    # Load datasets
    print("Loading training data...")
    train_examples = load_dataset_from_json(args.train_data)
    print(f"Loaded {len(train_examples)} training examples")
    
    # Create appropriate dataset based on task type
    if args.task_type == 'pretraining':
        # For pre-training, expect a list of raw texts
        texts = [example['text'] for example in train_examples]
        train_dataset = T5PretrainingDataset(
            texts, 
            tokenizer, 
            max_length=args.max_source_length
        )
    else:
        # For fine-tuning tasks
        if args.task_type != 'general':
            # Format examples for specific tasks
            train_examples = tokenizer.create_task_examples(args.task_type, train_examples)
        
        train_dataset = T5TextDataset(
            train_examples, 
            tokenizer, 
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            task_type=args.task_type
        )
    
    train_dataloader = create_dataloader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # Load evaluation data if provided
    eval_dataloader = None
    if args.eval_data:
        print("Loading evaluation data...")
        eval_examples = load_dataset_from_json(args.eval_data)
        print(f"Loaded {len(eval_examples)} evaluation examples")
        
        if args.task_type == 'pretraining':
            eval_texts = [example['text'] for example in eval_examples]
            eval_dataset = T5PretrainingDataset(
                eval_texts, 
                tokenizer, 
                max_length=args.max_source_length
            )
        else:
            if args.task_type != 'general':
                eval_examples = tokenizer.create_task_examples(args.task_type, eval_examples)
            
            eval_dataset = T5TextDataset(
                eval_examples, 
                tokenizer,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length,
                task_type=args.task_type
            )
        
        eval_dataloader = create_dataloader(
            eval_dataset, 
            batch_size=args.eval_batch_size, 
            shuffle=False
        )
    
    # Create training configuration
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_wandb=args.use_wandb,
        eval_during_training=eval_dataloader is not None
    )
    
    # Create trainer and start training
    trainer = T5Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )
    
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()