"""
Comprehensive unit tests for T5 model components
"""

import unittest
import torch
import torch.nn.functional as F
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.config import T5Config
from src.model.t5_model import T5ForConditionalGeneration, T5Model
from src.model.attention import T5Attention
from src.model.layers import T5LayerNorm, T5LayerFF, T5LayerSelfAttention
from src.model.encoder import T5Stack, T5Block
from src.model.decoder import T5DecoderStack, T5DecoderBlock


class TestT5Config(unittest.TestCase):
    """Test T5 configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = T5Config()
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.d_kv, 64)
        self.assertEqual(config.vocab_size, 32128)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = T5Config(d_model=768, num_heads=12, d_kv=64)
        self.assertEqual(config.d_kv, 64)  # Should auto-adjust
        
        # Invalid config - d_model not divisible by num_heads
        with self.assertRaises(ValueError):
            T5Config(d_model=100, num_heads=8)
    
    def test_config_serialization(self):
        """Test config to dict conversion"""
        config = T5Config(d_model=768, num_heads=12)
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['d_model'], 768)


class TestT5Layers(unittest.TestCase):
    """Test T5 layer components"""
    
    def setUp(self):
        self.config = T5Config(d_model=512, num_heads=8, d_kv=64, d_ff=2048)
        self.batch_size = 2
        self.seq_length = 10
    
    def test_layer_norm(self):
        """Test T5 LayerNorm"""
        layer_norm = T5LayerNorm(self.config.d_model)
        input_tensor = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        
        output = layer_norm(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, input_tensor.shape)
        
        # Check normalization (variance should be close to 1)
        variance = output.pow(2).mean(-1)
        expected_variance = torch.ones_like(variance)
        self.assertTrue(torch.allclose(variance, expected_variance, atol=1e-5))
    
    def test_feed_forward(self):
        """Test T5 feed-forward layer"""
        ff_layer = T5LayerFF(self.config)
        input_tensor = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        
        output = ff_layer(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, input_tensor.shape)
        
        # Check that output is different from input (transformation occurred)
        self.assertFalse(torch.allclose(output, input_tensor))
    
    def test_self_attention(self):
        """Test T5 self-attention layer"""
        attention_layer = T5LayerSelfAttention(self.config, has_relative_attention_bias=True)
        input_tensor = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        
        outputs = attention_layer(input_tensor)
        
        # Check output shape
        self.assertEqual(outputs[0].shape, input_tensor.shape)
        
        # Check that it's a tuple with at least one element
        self.assertIsInstance(outputs, tuple)
        self.assertGreaterEqual(len(outputs), 1)


class TestT5Attention(unittest.TestCase):
    """Test T5 attention mechanism"""
    
    def setUp(self):
        self.config = T5Config(d_model=512, num_heads=8, d_kv=64)
        self.batch_size = 2
        self.seq_length = 10
    
    def test_self_attention(self):
        """Test self-attention mechanism"""
        attention = T5Attention(self.config, has_relative_attention_bias=True)
        hidden_states = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        
        outputs = attention(hidden_states)
        
        # Check output shape
        self.assertEqual(outputs[0].shape, hidden_states.shape)
        
        # Check attention weights if output_attentions=True
        outputs_with_attn = attention(hidden_states, output_attentions=True)
        self.assertGreater(len(outputs_with_attn), 1)
    
    def test_cross_attention(self):
        """Test cross-attention mechanism"""
        attention = T5Attention(self.config, has_relative_attention_bias=False)
        hidden_states = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        key_value_states = torch.randn(self.batch_size, 15, self.config.d_model)  # Different length
        
        outputs = attention(hidden_states, key_value_states=key_value_states)
        
        # Check output shape matches query (hidden_states)
        self.assertEqual(outputs[0].shape, hidden_states.shape)
    
    def test_relative_position_bias(self):
        """Test relative position bias computation"""
        attention = T5Attention(self.config, has_relative_attention_bias=True)
        
        # Test bias computation
        query_length, key_length = 10, 12
        device = torch.device('cpu')
        
        bias = attention.compute_bias(query_length, key_length, device)
        
        # Check bias shape
        expected_shape = (1, self.config.num_heads, query_length, key_length)
        self.assertEqual(bias.shape, expected_shape)


class TestT5Blocks(unittest.TestCase):
    """Test T5 transformer blocks"""
    
    def setUp(self):
        self.config = T5Config(d_model=512, num_heads=8, d_kv=64, d_ff=2048, num_layers=2)
        self.batch_size = 2
        self.seq_length = 10
    
    def test_encoder_block(self):
        """Test T5 encoder block"""
        encoder_block = T5Block(self.config, has_relative_attention_bias=True)
        hidden_states = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        
        outputs = encoder_block(hidden_states)
        
        # Check output shape
        self.assertEqual(outputs[0].shape, hidden_states.shape)
    
    def test_decoder_block(self):
        """Test T5 decoder block"""
        decoder_block = T5DecoderBlock(self.config, has_relative_attention_bias=True)
        hidden_states = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        encoder_hidden_states = torch.randn(self.batch_size, 15, self.config.d_model)
        
        outputs = decoder_block(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states
        )
        
        # Check output shape
        self.assertEqual(outputs[0].shape, hidden_states.shape)


class TestT5Model(unittest.TestCase):
    """Test complete T5 model"""
    
    def setUp(self):
        self.config = T5Config(
            vocab_size=1000, 
            d_model=256, 
            num_heads=4, 
            d_kv=64, 
            d_ff=1024, 
            num_layers=2
        )
        self.batch_size = 2
        self.seq_length = 10
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = T5Model(self.config)
        
        # Check that model has required components
        self.assertTrue(hasattr(model, 'encoder'))
        self.assertTrue(hasattr(model, 'decoder'))
        self.assertTrue(hasattr(model, 'shared'))
        
        # Check shared embeddings
        self.assertEqual(model.shared.num_embeddings, self.config.vocab_size)
        self.assertEqual(model.shared.embedding_dim, self.config.d_model)
    
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        model = T5Model(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Check output format
        self.assertIsInstance(encoder_outputs, dict)
        self.assertIn('last_hidden_state', encoder_outputs)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_length, self.config.d_model)
        self.assertEqual(encoder_outputs['last_hidden_state'].shape, expected_shape)
    
    def test_full_model_forward(self):
        """Test full model forward pass"""
        model = T5Model(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        decoder_input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, 5))
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        # Check output format
        self.assertIsInstance(outputs, dict)
        self.assertIn('last_hidden_state', outputs)


class TestT5ForConditionalGeneration(unittest.TestCase):
    """Test T5 conditional generation model"""
    
    def setUp(self):
        self.config = T5Config(
            vocab_size=1000, 
            d_model=256, 
            num_heads=4, 
            d_kv=64, 
            d_ff=1024, 
            num_layers=2
        )
        self.batch_size = 2
        self.seq_length = 10
    
    def test_model_with_loss(self):
        """Test model forward pass with loss computation"""
        model = T5ForConditionalGeneration(self.config)
        
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        labels = torch.randint(0, self.config.vocab_size, (self.batch_size, 5))
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Check required outputs
        self.assertIn('loss', outputs)
        self.assertIn('logits', outputs)
        
        # Check loss properties
        self.assertTrue(outputs['loss'].requires_grad)
        self.assertGreater(outputs['loss'].item(), 0)
        
        # Check logits shape
        expected_logits_shape = (self.batch_size, 5, self.config.vocab_size)
        self.assertEqual(outputs['logits'].shape, expected_logits_shape)
    
    def test_generation(self):
        """Test text generation"""
        model = T5ForConditionalGeneration(self.config)
        model.eval()
        
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=15,
                do_sample=False
            )
        
        # Check generation output
        self.assertEqual(generated.shape[0], self.batch_size)
        self.assertLessEqual(generated.shape[1], 15)
    
    def test_gradient_flow(self):
        """Test gradient flow through model"""
        model = T5ForConditionalGeneration(self.config)
        
        input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        attention_mask = torch.ones(self.batch_size, self.seq_length)
        labels = torch.randint(0, self.config.vocab_size, (self.batch_size, 5))
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Backward pass
        outputs['loss'].backward()
        
        # Check gradients exist
        gradient_exists = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                gradient_exists = True
                break
        
        self.assertTrue(gradient_exists, "No gradients found after backward pass")


class TestT5Performance(unittest.TestCase):
    """Test T5 model performance characteristics"""
    
    def setUp(self):
        self.config = T5Config(
            vocab_size=1000, 
            d_model=512, 
            num_heads=8, 
            d_kv=64, 
            d_ff=2048, 
            num_layers=4
        )
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_compatibility(self):
        """Test CUDA compatibility"""
        device = torch.device("cuda")
        model = T5ForConditionalGeneration(self.config).to(device)
        
        batch_size, seq_length = 2, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, 5)).to(device)
        
        # Forward pass on GPU
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Check outputs are on GPU
        self.assertEqual(outputs['loss'].device.type, 'cuda')
        self.assertEqual(outputs['logits'].device.type, 'cuda')
    
    def test_memory_efficiency(self):
        """Test basic memory usage"""
        model = T5ForConditionalGeneration(self.config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # All params should be trainable by default
    
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths"""
        model = T5ForConditionalGeneration(self.config)
        
        batch_size = 2
        
        # Test different input lengths
        for seq_length in [5, 10, 20]:
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
            attention_mask = torch.ones(batch_size, seq_length)
            labels = torch.randint(0, self.config.vocab_size, (batch_size, 3))
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            self.assertIn('loss', outputs)
            self.assertIn('logits', outputs)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)