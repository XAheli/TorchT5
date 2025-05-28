"""
T5 Encoder Implementation
Implements the bidirectional encoder stack with self-attention and feed-forward layers
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .layers import T5LayerSelfAttention, T5LayerFF, T5LayerNorm


class T5Block(nn.Module):
    """
    T5 Transformer Block for Encoder
    Contains self-attention and feed-forward layers with residual connections
    """
    
    def __init__(self, config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # Self-attention - call layer[0] explicitly
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
            layer_head_mask=layer_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs[0]
        
        if output_attentions:
            attention_weights = self_attention_outputs[1]
        
        # Feed-forward - call layer[1] explicitly
        hidden_states = self.layer[1](hidden_states)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        
        # Return position bias for sharing (only from first layer)
        if self.layer[0].SelfAttention.has_relative_attention_bias:
            outputs += (self_attention_outputs[-1],)  # Position bias
        
        return outputs


class T5Stack(nn.Module):
    """
    T5 Encoder Stack
    Contains multiple T5Block layers with embedding and final layer norm
    """
    
    def __init__(self, config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        
        self.embed_tokens = embed_tokens
        self.is_decoder = False
        
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # Attention mask processing
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)

        # For encoder, we use bidirectional attention (no causal masking)
        extended_attention_mask = self.invert_attention_mask(attention_mask)

        # Initialize states
        hidden_states = self.dropout(inputs_embeds)
        position_bias = None  # Initialize position bias
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,  # Pass existing bias
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # Extract position bias from first layer and reuse
            if i == 0:
                # Extract position bias from first layer for reuse
                position_bias = layer_outputs[-1]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }

    def invert_attention_mask(self, attention_mask: torch.Tensor):
        """Convert 1s and 0s attention mask to additive mask with large negative values"""
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        return extended_attention_mask

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        for block in self.block:
            for layer in block.layer:
                if hasattr(layer, 'SelfAttention'):
                    layer.SelfAttention.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        for block in self.block:
            for layer in block.layer:
                if hasattr(layer, 'SelfAttention'):
                    layer.SelfAttention.gradient_checkpointing = False
