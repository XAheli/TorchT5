"""
Simple T5 training example
Demonstrates basic fine-tuning on a small dataset
"""

import torch
import json
import os
from torch.utils.data import DataLoader

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.config import T5Config
from src.model.t5_model import T5ForConditionalGeneration
from src.data.tokenizer import T5Tokenizer, download_t5_tokenizer
from src.data.dataset import T5TextDataset
from src.training.trainer import TrainingConfig, T5Trainer


def create_sample_data():
    """Create sample training data for demonstration"""
    sample_data = [
        {
            "input_text": "translate English to German: Hello, how are you?",
            "target_text": "Hallo, wie geht es dir?"
        },
        {
            "input_text": "translate English to German: Good morning!",
            "target_text": "Guten Morgen!"
        },
        {
            "input_text": "translate English to German: Thank you very much.",
            "target_text": "Vielen Dank."
        },
        {
            "input_text": "summarize: The quick brown fox jumps over the lazy dog. This sentence contains all letters of the alphabet.",
            "target_text": "A sentence with all alphabet letters featuring a fox and dog."
        },
        {
            "input_text": "summarize: Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "target_text": "ML is AI subset enabling computer learning without explicit programming."
        },
        {
            "input_text": "question: What is the capital of France?",
            "target_text": "Paris"
        },
        {
            "input_text": "question: How many days are in a week?",
            "target_text": "Seven"
        },
        {
            "input_text": "sentiment: I love this movie! It's fantastic.",
            "target_text": "positive"
        },
        {
            "input_text": "sentiment: This product is terrible and doesn't work.",
            "target_text": "negative"
        },
        {
            "input_text": "sentiment: The weather is okay today.",
            "target_text": "neutral"
        }
    ]
    
    return sample_data


def main():
    """Main training function"""
    print(" Starting T5 Simple Training Example")
    
    # Configuration
    output_dir = "./simple_training_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Model configuration (small for quick training)
    model_config = T5Config(
        vocab_size=32128,  # Will be updated after loading tokenizer
        d_model=256,       # Smaller for faster training
        d_ff=1024,
        num_layers=4,      # Fewer layers
        num_heads=8,
        dropout_rate=0.1
    )
    
    # Training configuration
    training_config = TrainingConfig(
        learning_rate=5e-4,
        num_epochs=10,
        batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_steps=50,
        eval_steps=25,
        logging_steps=10,
        output_dir=output_dir,
        use_wandb=False,  # Set to True if you want to use wandb
        eval_during_training=True
    )
    
    print(" Setting up tokenizer...")
    # Download and setup tokenizer
    tokenizer_path = download_t5_tokenizer("t5-small")
    tokenizer = T5Tokenizer(tokenizer_path)
    
    # Update model config with actual vocab size
    model_config.vocab_size = tokenizer.get_vocab_size()
    print(f" Tokenizer loaded with vocab size: {model_config.vocab_size}")
    
    print(" Creating model...")
    # Create model
    model = T5ForConditionalGeneration(model_config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model created with {total_params:,} parameters")
    print(f" Trainable parameters: {trainable_params:,}")
    
    print(" Preparing datasets...")
    # Create sample data
    sample_data = create_sample_data()
    
    # Split data for training and validation
    train_data = sample_data[:8]
    eval_data = sample_data[8:]
    
    # Create datasets
    train_dataset = T5TextDataset(
        train_data,
        tokenizer,
        max_source_length=128,
        max_target_length=32,
        task_type="general"
    )
    
    eval_dataset = T5TextDataset(
        eval_data,
        tokenizer,
        max_source_length=128,
        max_target_length=32,
        task_type="general"
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    print(f" Training set size: {len(train_dataset)}")
    print(f" Evaluation set size: {len(eval_dataset)}")
    
    print(" Starting training...")
    # Create trainer
    trainer = T5Trainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )
    
    # Start training
    trainer.train()
    
    print(" Training completed!")
    print(f" Model saved to: {output_dir}")
    
    # Test the trained model
    print("\n Testing trained model...")
    test_inference(model, tokenizer)


def test_inference(model, tokenizer):
    """Test the trained model with some examples"""
    from src.inference.generator import T5Generator, GenerationConfig
    
    model.eval()
    generator = T5Generator(model, tokenizer)
    
    generation_config = GenerationConfig(
        max_length=32,
        num_beams=2,
        early_stopping=True,
        temperature=1.0
    )
    
    test_inputs = [
        "translate English to German: Good evening!",
        "summarize: Artificial intelligence is transforming the world.",
        "sentiment: This is an amazing product!",
        "question: What is 2 + 2?"
    ]
    
    print("\n--- Test Results ---")
    for i, input_text in enumerate(test_inputs, 1):
        try:
            output = generator.generate([input_text], generation_config)[0]
            print(f"{i}. Input: {input_text}")
            print(f"   Output: {output}\n")
        except Exception as e:
            print(f"{i}. Input: {input_text}")
            print(f"   Error: {e}\n")


if __name__ == "__main__":
    main()