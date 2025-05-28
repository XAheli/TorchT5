"""
T5 Core Layers Implementation
Implements LayerNorm, FeedForward, and other essential components following T5 paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class T5LayerNorm(nn.Module):
    """
    T5-style LayerNorm without bias, applied before transformations (pre-norm)
    Based on: "Layer normalization is applied only to the input of each sub-layer"
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Only weight, no bias
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # T5 uses Root Mean Square Layer Normalization
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class T5DenseActDense(nn.Module):
    """
    T5 Feed-Forward Network with ReLU activation
    Architecture: Linear -> ReLU -> Dropout -> Linear
    """
    
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    """
    T5 Feed-Forward Layer with pre-norm and residual connection
    """
    
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5LayerSelfAttention(nn.Module):
    """
    T5 Self-Attention Layer with pre-norm and residual connection
    """
    
    def __init__(self, config, has_relative_attention_bias: bool = False):
        super().__init__()
        from .attention import T5Attention
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
            layer_head_mask=layer_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # Properly handle attention output
        if output_attentions:
            attention_weights = attention_output[-1]
            attention_output = attention_output[0]
        else:
            attention_output = attention_output[0]
        
        hidden_states = hidden_states + self.dropout(attention_output)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs


class T5LayerCrossAttention(nn.Module):
    """
    T5 Cross-Attention Layer for encoder-decoder attention
    """
    
    def __init__(self, config):
        super().__init__()
        from .attention import T5Attention
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            layer_head_mask=layer_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # Properly handle attention output
        if output_attentions:
            attention_weights = attention_output[-1]
            attention_output = attention_output[0]
        else:
            attention_output = attention_output[0]
        
        hidden_states = hidden_states + self.dropout(attention_output)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs
