"""
Detailed tests for T5 attention mechanisms
"""

import unittest
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.config import T5Config
from src.model.attention import T5Attention


class TestT5AttentionMechanisms(unittest.TestCase):
    """Comprehensive tests for T5 attention"""
    
    def setUp(self):
        self.config = T5Config(
            d_model=512,
            num_heads=8,
            d_kv=64,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128
        )
        self.batch_size = 2
        self.seq_length = 12
        self.device = torch.device('cpu')
    
    def test_relative_position_bucketing(self):
        """Test relative position bucketing logic"""
        attention = T5Attention(self.config, has_relative_attention_bias=True)
        
        # Test bidirectional bucketing (encoder)
        relative_position = torch.tensor([[-2, -1, 0, 1, 2], [-3, -1, 0, 2, 4]])
        buckets = attention._relative_position_bucket(
            relative_position, 
            bidirectional=True,
            num_buckets=32,
            max_distance=128
        )
        
        # Check output shape
        self.assertEqual(buckets.shape, relative_position.shape)
        
        # Check all buckets are valid
        self.assertTrue((buckets >= 0).all())
        self.assertTrue((buckets < 32).all())
        
        # Test unidirectional bucketing (decoder)
        buckets_unidirectional = attention._relative_position_bucket(
            relative_position,
            bidirectional=False,
            num_buckets=32,
            max_distance=128
        )
        
        self.assertEqual(buckets_unidirectional.shape, relative_position.shape)
    
    def test_attention_bias_computation(self):
        """Test attention bias computation"""
        attention = T5Attention(self.config, has_relative_attention_bias=True)
        
        query_length, key_length = 8, 10
        bias = attention.compute_bias(query_length, key_length, self.device)
        
        # Check bias shape
        expected_shape = (1, self.config.num_heads, query_length, key_length)
        self.assertEqual(bias.shape, expected_shape)
        
        # Check bias is finite
        self.assertTrue(torch.isfinite(bias).all())
    
    def test_attention_mask_application(self):
        """Test attention mask application"""
        attention = T5Attention(self.config, has_relative_attention_bias=True)
        
        hidden_states = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        
        # Create attention mask (mask out last few tokens)
        attention_mask = torch.ones(self.batch_size, 1, 1, self.seq_length)
        attention_mask[:, :, :, -2:] = 0  # Mask last 2 tokens
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        
        outputs = attention(hidden_states, mask=attention_mask, output_attentions=True)
        
        # Check that attention weights for masked positions are very small
        attention_weights = outputs[-1]  # Last output is attention weights
        masked_attention = attention_weights[:, :, :, -2:]  # Last 2 positions
        
        # Attention weights for masked positions should be close to zero
        self.assertTrue((masked_attention < 1e-6).all())
    
    def test_causal_attention_pattern(self):
        """Test causal attention pattern for decoder"""
        # Create decoder attention
        config = self.config
        config.is_decoder = True
        attention = T5Attention(config, has_relative_attention_bias=True)
        
        hidden_states = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        
        # Create causal mask
        seq_ids = torch.arange(self.seq_length)
        causal_mask = seq_ids[None, :] <= seq_ids[:, None]
        causal_mask = causal_mask.float().unsqueeze(0).unsqueeze(0)
        causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float32).min
        
        outputs = attention(hidden_states, mask=causal_mask, output_attentions=True)
        attention_weights = outputs[-1]
        
        # Check causal pattern: attention_weights[i, j] should be ~0 for j > i
        for head in range(self.config.num_heads):
            for i in range(self.seq_length):
                for j in range(i + 1, self.seq_length):
                    self.assertLess(
                        attention_weights[0, head, i, j].item(), 1e-6,
                        f"Causal constraint violated at position ({i}, {j})"
                    )
    
    def test_cross_attention(self):
        """Test cross-attention between encoder and decoder"""
        attention = T5Attention(self.config, has_relative_attention_bias=False)
        
        decoder_hidden_states = torch.randn(self.batch_size, 8, self.config.d_model)
        encoder_hidden_states = torch.randn(self.batch_size, 12, self.config.d_model)
        
        outputs = attention(
            decoder_hidden_states,
            key_value_states=encoder_hidden_states,
            output_attentions=True
        )
        
        # Check output shape matches decoder input
        self.assertEqual(outputs[0].shape, decoder_hidden_states.shape)
        
        # Check attention weights shape
        attention_weights = outputs[-1]
        expected_attn_shape = (self.batch_size, self.config.num_heads, 8, 12)
        self.assertEqual(attention_weights.shape, expected_attn_shape)
        
        # Check attention weights sum to 1 across key dimension
        attn_sums = attention_weights.sum(dim=-1)
        expected_sums = torch.ones_like(attn_sums)
        self.assertTrue(torch.allclose(attn_sums, expected_sums, atol=1e-5))
    
    def test_attention_gradients(self):
        """Test gradient flow through attention"""
        attention = T5Attention(self.config, has_relative_attention_bias=True)
        
        hidden_states = torch.randn(
            self.batch_size, self.seq_length, self.config.d_model, 
            requires_grad=True
        )
        
        outputs = attention(hidden_states)
        loss = outputs[0].sum()
        loss.backward()
        
        # Check gradients exist and are finite
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())
        
        # Check attention layer gradients
        for name, param in attention.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}")
    
    def test_attention_output_consistency(self):
        """Test attention output consistency across multiple runs"""
        attention = T5Attention(self.config, has_relative_attention_bias=True)
        attention.eval()  # Set to eval mode for deterministic behavior
        
        hidden_states = torch.randn(self.batch_size, self.seq_length, self.config.d_model)
        
        with torch.no_grad():
            outputs1 = attention(hidden_states)
            outputs2 = attention(hidden_states)
        
        # Outputs should be identical in eval mode
        self.assertTrue(torch.allclose(outputs1[0], outputs2[0]))
    
    def test_different_head_configurations(self):
        """Test attention with different head configurations"""
        for num_heads in [1, 4, 8, 16]:
            if 512 % num_heads == 0:  # Valid configuration
                config = T5Config(d_model=512, num_heads=num_heads, d_kv=512 // num_heads)
                attention = T5Attention(config, has_relative_attention_bias=True)
                
                hidden_states = torch.randn(self.batch_size, self.seq_length, 512)
                outputs = attention(hidden_states)
                
                # Check output shape is preserved
                self.assertEqual(outputs[0].shape, hidden_states.shape)
    
    def test_attention_with_past_key_values(self):
        """Test attention with cached key-value states"""
        attention = T5Attention(self.config, has_relative_attention_bias=True)
        
        # Initial forward pass
        hidden_states1 = torch.randn(self.batch_size, 5, self.config.d_model)
        outputs1 = attention(hidden_states1, use_cache=True)
        
        # Get past key-value states
        past_key_value = outputs1[1] if len(outputs1) > 1 else None
        
        if past_key_value is not None:
            # Next step with cached states
            hidden_states2 = torch.randn(self.batch_size, 1, self.config.d_model)
            outputs2 = attention(
                hidden_states2, 
                past_key_value=past_key_value,
                use_cache=True
            )
            
            # Check output shape
            self.assertEqual(outputs2[0].shape, hidden_states2.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)