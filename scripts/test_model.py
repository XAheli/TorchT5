"""
Comprehensive test script for T5 model components
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.config import T5Config
from src.model.t5_model import T5ForConditionalGeneration, T5Model
from src.model.attention import T5Attention
from src.model.layers import T5LayerNorm, T5LayerFF
from src.training.utils import count_parameters, get_model_size_mb


def test_config():
    """Test T5 configuration"""
    print("Testing T5Config...")
    
    config = T5Config()
    assert config.d_model == 512
    assert config.num_heads == 8
    assert config.d_kv == 64
    
    # Test validation
    try:
        invalid_config = T5Config(d_model=100, num_heads=8)  # Should fail
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print(" T5Config tests passed")


def test_layer_norm():
    """Test T5 LayerNorm"""
    print("Testing T5LayerNorm...")
    
    layer_norm = T5LayerNorm(512)
    input_tensor = torch.randn(2, 10, 512)
    
    output = layer_norm(input_tensor)
    assert output.shape == input_tensor.shape
    
    # Test that it normalizes (variance should be close to 1)
    variance = output.pow(2).mean(-1)
    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-5)
    
    print(" T5LayerNorm tests passed")


def test_attention():
    """Test T5 attention mechanism"""
    print("Testing T5Attention...")
    
    config = T5Config(d_model=512, num_heads=8, d_kv=64)
    attention = T5Attention(config, has_relative_attention_bias=True)
    
    batch_size, seq_length = 2, 10
    hidden_states = torch.randn(batch_size, seq_length, config.d_model)
    
    outputs = attention(hidden_states)
    
    assert len(outputs) >= 1
    assert outputs[0].shape == hidden_states.shape
    
    print(" T5Attention tests passed")


def test_feed_forward():
    """Test T5 feed-forward layer"""
    print("Testing T5LayerFF...")
    
    config = T5Config(d_model=512, d_ff=2048)
    ff_layer = T5LayerFF(config)
    
    batch_size, seq_length = 2, 10
    hidden_states = torch.randn(batch_size, seq_length, config.d_model)
    
    output = ff_layer(hidden_states)
    assert output.shape == hidden_states.shape
    
    print(" T5LayerFF tests passed")


def test_encoder_decoder():
    """Test T5 encoder and decoder"""
    print("Testing T5 encoder and decoder...")
    
    config = T5Config(vocab_size=1000, d_model=512, num_layers=2)
    model = T5Model(config)
    
    batch_size, seq_length = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Test encoder only
    encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    assert 'last_hidden_state' in encoder_outputs
    assert encoder_outputs['last_hidden_state'].shape == (batch_size, seq_length, config.d_model)
    
    # Test full model
    decoder_input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids
    )
    
    assert 'last_hidden_state' in outputs
    
    print(" T5Model tests passed")


def test_conditional_generation():
    """Test T5ForConditionalGeneration"""
    print("Testing T5ForConditionalGeneration...")
    
    config = T5Config(vocab_size=1000, d_model=512, num_layers=2)
    model = T5ForConditionalGeneration(config)
    
    batch_size, seq_length = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, config.vocab_size, (batch_size, 5))
    
    # Test forward pass with loss computation
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    assert 'loss' in outputs
    assert 'logits' in outputs
    assert outputs['loss'].requires_grad
    assert outputs['logits'].shape == (batch_size, 5, config.vocab_size)
    
    # Test generation
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=15
    )
    
    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= 15
    
    print(" T5ForConditionalGeneration tests passed")


def test_model_parameters():
    """Test model parameter counting and sizing"""
    print("Testing model parameter utilities...")
    
    config = T5Config(d_model=512, num_layers=6)
    model = T5ForConditionalGeneration(config)
    
    param_info = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    
    assert 'total_parameters' in param_info
    assert 'trainable_parameters' in param_info
    assert param_info['total_parameters'] > 0
    assert param_info['trainable_parameters'] > 0
    assert model_size_mb > 0
    
    print(f" Model has {param_info['total_parameters']:,} parameters ({model_size_mb:.2f} MB)")


def test_gradient_flow():
    """Test gradient flow through the model"""
    print("Testing gradient flow...")
    
    config = T5Config(vocab_size=1000, d_model=256, num_layers=2, num_heads=4)
    model = T5ForConditionalGeneration(config)
    
    batch_size, seq_length = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, config.vocab_size, (batch_size, 5))
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs['loss']
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    gradient_exists = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            gradient_exists = True
            break
    
    assert gradient_exists, "No gradients found"
    
    print(" Gradient flow test passed")


def test_memory_efficiency():
    """Test model memory usage"""
    print("Testing memory efficiency...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
        config = T5Config(vocab_size=1000, d_model=512, num_layers=4)
        model = T5ForConditionalGeneration(config).to(device)
        
        batch_size, seq_length = 4, 20
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        labels = torch.randint(0, config.vocab_size, (batch_size, 10)).to(device)
        
        # Measure memory before
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Measure memory after
        memory_after = torch.cuda.memory_allocated()
        memory_used_mb = (memory_after - memory_before) / 1024 / 1024
        
        print(f" Memory usage test passed ({memory_used_mb:.2f} MB used)")
    else:
        print(" Memory test skipped (no CUDA available)")


def run_all_tests():
    """Run all tests"""
    print("Running T5 model tests...\n")
    
    try:
        test_config()
        test_layer_norm()
        test_attention()
        test_feed_forward()
        test_encoder_decoder()
        test_conditional_generation()
        test_model_parameters()
        test_gradient_flow()
        test_memory_efficiency()
        
        print("\n All tests passed!")
        
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()