"""
Advanced text generation example with T5
Demonstrates various generation strategies and tasks
"""

import torch
import argparse
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.config import T5Config
from src.model.t5_model import T5ForConditionalGeneration
from src.data.tokenizer import T5Tokenizer, download_t5_tokenizer
from src.inference.generator import T5Generator, GenerationConfig
from src.training.utils import load_checkpoint


def create_model_and_tokenizer(model_size="small"):
    """Create T5 model and tokenizer"""
    print(f" Creating T5-{model_size} model...")
    
    # Model configurations
    size_configs = {
        "small": {"d_model": 512, "d_ff": 2048, "num_layers": 6, "num_heads": 8},
        "base": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    }
    
    # Download tokenizer
    tokenizer_path = download_t5_tokenizer(f"t5-{model_size}")
    tokenizer = T5Tokenizer(tokenizer_path)
    
    # Create model
    config = T5Config(**size_configs[model_size])
    config.vocab_size = tokenizer.get_vocab_size()
    
    model = T5ForConditionalGeneration(config)
    
    return model, tokenizer


def demonstrate_translation(generator):
    """Demonstrate translation capabilities"""
    print("\n Translation Examples")
    print("=" * 50)
    
    examples = [
        ("Hello, how are you today?", "English", "German"),
        ("I love machine learning.", "English", "French"),
        ("The weather is beautiful.", "English", "Spanish"),
    ]
    
    for text, src_lang, tgt_lang in examples:
        result = generator.translate(text, src_lang, tgt_lang, max_length=50)
        print(f"   {src_lang} → {tgt_lang}")
        print(f"   Input: {text}")
        print(f"   Output: {result}\n")


def demonstrate_summarization(generator):
    """Demonstrate summarization capabilities"""
    print("\n Summarization Examples")
    print("=" * 50)
    
    texts = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
        
        "Climate change refers to long-term shifts in global or regional climate patterns. Since the mid-20th century, humans have been the dominant driver of climate change, primarily due to fossil fuel burning which increases heat-trapping greenhouse gas levels in Earth's atmosphere.",
        
        "The Internet is a global system of interconnected computer networks that uses the Internet protocol suite to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope."
    ]
    
    for i, text in enumerate(texts, 1):
        summary = generator.summarize(text, max_length=50)
        print(f"   Example {i}")
        print(f"   Text: {text[:100]}...")
        print(f"   Summary: {summary}\n")


def demonstrate_question_answering(generator):
    """Demonstrate question answering capabilities"""
    print("\n Question Answering Examples")
    print("=" * 50)
    
    qa_pairs = [
        ("What is the capital of France?", "France is a country in Europe. Its capital city is Paris, which is also its largest city."),
        ("How many planets are in our solar system?", "Our solar system consists of eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.")
    ]
    
    for question, context in qa_pairs:
        answer = generator.answer_question(question, context, max_length=30)
        print(f" Question: {question}")
        print(f" Context: {context[:100]}...")
        print(f" Answer: {answer}\n")


def demonstrate_generation_strategies(generator, input_text):
    """Demonstrate different generation strategies"""
    print("\n Generation Strategy Comparison")
    print("=" * 50)
    print(f"Input: {input_text}\n")
    
    strategies = [
        ("Greedy", GenerationConfig(max_length=50, do_sample=False, num_beams=1)),
        ("Beam Search", GenerationConfig(max_length=50, do_sample=False, num_beams=4)),
        ("Top-k Sampling", GenerationConfig(max_length=50, do_sample=True, top_k=50, temperature=0.8)),
        ("Top-p Sampling", GenerationConfig(max_length=50, do_sample=True, top_p=0.9, temperature=0.8)),
        ("Temperature 0.5", GenerationConfig(max_length=50, do_sample=True, temperature=0.5)),
        ("Temperature 1.5", GenerationConfig(max_length=50, do_sample=True, temperature=1.5)),
    ]
    
    for name, config in strategies:
        try:
            output = generator.generate([input_text], config)[0]
            print(f"🔧 {name}: {output}")
        except Exception as e:
            print(f"🔧 {name}: Error - {e}")
    
    print()


def interactive_mode(generator):
    """Interactive generation mode"""
    print("\n Interactive Mode")
    print("=" * 50)
    print("Enter your prompts (type 'quit' to exit):")
    print("Examples:")
    print("  - translate English to French: Hello world")
    print("  - summarize: [your text]")
    print("  - question: What is AI? context: [context]")
    print()
    
    while True:
        try:
            user_input = input(" Prompt: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Generation configuration
            config = GenerationConfig(
                max_length=100,
                num_beams=2,
                temperature=0.8,
                do_sample=True,
                top_p=0.9
            )
            
            output = generator.generate([user_input], config)[0]
            print(f" Output: {output}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f" Error: {e}\n")
    
    print(" Goodbye!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="T5 Text Generation Examples")
    parser.add_argument("--model_size", choices=["small", "base"], default="small",
                       help="T5 model size")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--demo", choices=["translation", "summarization", "qa", "strategies", "all"],
                       default="all", help="Which demo to run")
    
    args = parser.parse_args()
    
    print(" T5 Text Generation Demo")
    print("=" * 50)
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(args.model_size)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f" Loading checkpoint from {args.checkpoint}")
        checkpoint_data = load_checkpoint(args.checkpoint, model)
        if checkpoint_data:
            print(" Checkpoint loaded successfully")
        else:
            print(" Failed to load checkpoint")
    
    # Create generator
    generator = T5Generator(model, tokenizer)
    
    # Run demonstrations
    if args.interactive:
        interactive_mode(generator)
    else:
        if args.demo in ["translation", "all"]:
            demonstrate_translation(generator)
        
        if args.demo in ["summarization", "all"]:
            demonstrate_summarization(generator)
        
        if args.demo in ["qa", "all"]:
            demonstrate_question_answering(generator)
        
        if args.demo in ["strategies", "all"]:
            demonstrate_generation_strategies(
                generator, 
                "translate English to German: The future of AI is bright."
            )


if __name__ == "__main__":
    main()