"""
T5 Decoder Implementation  
Implements the autoregressive decoder stack with self-attention, cross-attention, and feed-forward layers
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .layers import T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF, T5LayerNorm


class T5DecoderBlock(nn.Module):
    """
    T5 Transformer Block for Decoder
    Contains self-attention, cross-attention, and feed-forward layers
    """
    
    def __init__(self, config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # Self-attention - call layer[0] explicitly
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=self_attn_past_key_value,
            layer_head_mask=layer_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs[0]
        
        present_key_value = self_attention_outputs[1] if use_cache else None
        
        if output_attentions:
            self_attention_weights = self_attention_outputs[-1]

        # Cross-attention - call layer[1] explicitly
        if encoder_hidden_states is not None:
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                past_key_value=cross_attn_past_key_value,
                layer_head_mask=cross_attn_layer_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]
            
            if use_cache:
                present_key_value = present_key_value + (cross_attention_outputs[1],)
            
            if output_attentions:
                cross_attention_weights = cross_attention_outputs[-1]

        # Feed-forward - call layer[2] explicitly
        hidden_states = self.layer[2](hidden_states)

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (present_key_value,)
        if output_attentions:
            outputs = outputs + (self_attention_weights,)
            if encoder_hidden_states is not None:
                outputs = outputs + (cross_attention_weights,)

        # Return position biases for sharing (only from first layer)
        if self.layer[0].SelfAttention.has_relative_attention_bias:
            outputs = outputs + (self_attention_outputs[-1],)  # Self-attention position bias
            if encoder_hidden_states is not None:
                outputs = outputs + (cross_attention_outputs[-1],)  # Cross-attention position bias

        return outputs


class T5DecoderStack(nn.Module):
    """
    T5 Decoder Stack
    Contains multiple T5DecoderBlock layers with embedding and final layer norm
    """
    
    def __init__(self, config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        
        self.embed_tokens = embed_tokens
        self.is_decoder = True
        
        self.block = nn.ModuleList(
            [T5DecoderBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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

        # Attention mask processing for causal attention
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device)

        # Create causal mask for decoder self-attention
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # Encoder attention mask processing
        if encoder_hidden_states is not None:
            if encoder_attention_mask is None:
                encoder_seq_length = encoder_hidden_states.shape[1]  # Proper indexing
                encoder_attention_mask = torch.ones(
                    batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Initialize states
        hidden_states = self.dropout(inputs_embeds)
        position_bias = None  # Initialize position bias
        encoder_decoder_position_bias = None  # Initialize cross-attention bias
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        present_key_value_states = () if use_cache else None

        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,  # Pass existing bias
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,  # Pass existing cross-attn bias
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            # Extract and reuse position biases from first layer
            if i == 0:
                position_bias = layer_outputs[-1]
                if encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[-2]

            if use_cache:
                present_key_value_states = present_key_value_states + (layer_outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)
                if encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[3],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            'last_hidden_state': hidden_states,
            'past_key_values': present_key_value_states,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions,
            'cross_attentions': all_cross_attentions,
        }

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int], device: torch.device):
        """Create causal mask for decoder self-attention"""
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(torch.long)

        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]

        # Combine causal mask with attention mask
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        return extended_attention_mask

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
                if hasattr(layer, 'EncDecAttention'):
                    layer.EncDecAttention.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        for block in self.block:
            for layer in block.layer:
                if hasattr(layer, 'SelfAttention'):
                    layer.SelfAttention.gradient_checkpointing = False
                if hasattr(layer, 'EncDecAttention'):
                    layer.EncDecAttention.gradient_checkpointing = False