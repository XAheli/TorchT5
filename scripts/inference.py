"""
Inference script for T5 model
Supports various text generation tasks
"""

import argparse
import torch
import json
import os

from src.model.config import T5Config
from src.model.t5_model import T5ForConditionalGeneration
from src.data.tokenizer import T5Tokenizer
from src.inference.generator import T5Generator, GenerationConfig
from src.training.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with T5 model")
    
    # Model arguments
    parser.add_argument('--tokenizer_model', type=str, required=True, help='Path to SentencePiece model file')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--config_file', type=str, help='Path to model config file')
    
    # Input arguments
    parser.add_argument('--input_text', type=str, help='Input text for generation')
    parser.add_argument('--input_file', type=str, help='Path to file containing input texts (one per line)')
    parser.add_argument('--task_type', type=str, default='general', 
                       choices=['general', 'translation', 'summarization', 'qa'])
    
    # Task-specific arguments
    parser.add_argument('--source_lang', type=str, default='English', help='Source language for translation')
    parser.add_argument('--target_lang', type=str, default='German', help='Target language for translation')
    parser.add_argument('--question', type=str, help='Question for QA task')
    parser.add_argument('--context', type=str, help='Context for QA task')
    
    # Generation arguments
    parser.add_argument('--max_length', type=int, default=50, help='Maximum generation length')
    parser.add_argument('--min_length', type=int, default=1, help='Minimum generation length')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus) sampling')
    parser.add_argument('--do_sample', action='store_true', help='Use sampling instead of greedy decoding')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='Length penalty for beam search')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, help='Path to output file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, tokenizer: T5Tokenizer, config_file: str = None):
    """Load T5 model from checkpoint"""
    # Load config
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        config = T5Config(**config_dict)
    else:
        # Try to load config from checkpoint directory
        config_path = os.path.join(checkpoint_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = T5Config(**config_dict)
        else:
            # Use default config
            print("Warning: No config file found, using default configuration")
            config = T5Config(vocab_size=tokenizer.get_vocab_size())
    
    # Create model
    model = T5ForConditionalGeneration(config)
    
    # Load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path, model)
    if checkpoint_data is None:
        raise ValueError(f"Could not load checkpoint from {checkpoint_path}")
    
    model.eval()
    return model


def prepare_input_text(args, text: str) -> str:
    """Prepare input text based on task type"""
    if args.task_type == 'translation':
        return f"translate {args.source_lang} to {args.target_lang}: {text}"
    elif args.task_type == 'summarization':
        return f"summarize: {text}"
    elif args.task_type == 'qa':
        if args.context:
            return f"question: {text} context: {args.context}"
        else:
            return f"question: {text}"
    else:
        return text


def main():
    args = parse_args()
    
    # Load tokenizer
    if not os.path.exists(args.tokenizer_model):
        raise FileNotFoundError(f"Tokenizer model not found: {args.tokenizer_model}")
    
    tokenizer = T5Tokenizer(args.tokenizer_model)
    print(f"Loaded tokenizer with vocab size: {tokenizer.get_vocab_size()}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_checkpoint, tokenizer, args.config_file)
    print("Model loaded successfully")
    
    # Create generator
    generation_config = GenerationConfig(
        max_length=args.max_length,
        min_length=args.min_length,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty
    )
    
    generator = T5Generator(model, tokenizer)
    
    # Prepare input texts
    input_texts = []
    
    if args.input_text:
        input_texts = [prepare_input_text(args, args.input_text)]
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            raw_texts = [line.strip() for line in f if line.strip()]
        input_texts = [prepare_input_text(args, text) for text in raw_texts]
    elif args.task_type == 'qa' and args.question:
        input_texts = [prepare_input_text(args, args.question)]
    else:
        raise ValueError("Must provide either --input_text, --input_file, or --question for QA task")
    
    print(f"Processing {len(input_texts)} input(s)...")
    
    # Generate outputs
    all_outputs = []
    
    # Process in batches
    for i in range(0, len(input_texts), args.batch_size):
        batch_inputs = input_texts[i:i + args.batch_size]
        
        print(f"Generating for batch {i // args.batch_size + 1}...")
        outputs = generator.generate(batch_inputs, generation_config)
        all_outputs.extend(outputs)
    
    # Display/save results
    for i, (input_text, output_text) in enumerate(zip(input_texts, all_outputs)):
        print(f"\n--- Example {i + 1} ---")
        print(f"Input: {input_text}")
        print(f"Output: {output_text}")
    
    # Save to file if specified
    if args.output_file:
        results = []
        for input_text, output_text in zip(input_texts, all_outputs):
            results.append({
                'input': input_text,
                'output': output_text
            })
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()