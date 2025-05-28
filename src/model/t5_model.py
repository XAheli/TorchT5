import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
import warnings

from .config import T5Config
from .encoder import T5Stack
from .decoder import T5DecoderStack
from .layers import T5LayerNorm


class T5Model(nn.Module):
    """
    Complete T5 Model with Encoder and Decoder
    Implements the text-to-text transfer transformer architecture
    """
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = config
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = config 
        decoder_config.is_decoder = True
        self.decoder = T5DecoderStack(decoder_config, self.shared)

        self.init_weights()

    def init_weights(self):
        """Initialize weights following T5 initialization scheme"""
        # Apply the initialization to all submodules
        def _init_weights(module):
            """Initialize the weights"""
            factor = self.config.initializer_factor  # Used for testing weights initialization
            if isinstance(module, T5LayerNorm):
                module.weight.data.fill_(1.0)
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=factor * 1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, T5Model):
                # Mesh TensorFlow embeddings initialization
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
                module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)

        self.apply(_init_weights)

        # Tie weights if specified
        if self.config.tie_word_embeddings:
            self._tie_weights()

    def _tie_weights(self):
        """Tie the word embeddings with the input embeddings"""
        if hasattr(self, "lm_head"):
            self._tie_or_clone_weights(self.lm_head, self.shared)

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        output_embeddings.weight = input_embeddings.weight

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        # Encode if encoder_outputs are not provided
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, dict):
            encoder_outputs = {
                'last_hidden_state': encoder_outputs[0],
                'hidden_states': encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                'attentions': encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            }

        hidden_states = encoder_outputs['last_hidden_state']

        # Decode
        if decoder_input_ids is not None or decoder_inputs_embeds is not None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            return {
                'last_hidden_state': decoder_outputs['last_hidden_state'],
                'past_key_values': decoder_outputs['past_key_values'],
                'decoder_hidden_states': decoder_outputs['hidden_states'],
                'decoder_attentions': decoder_outputs['attentions'],
                'cross_attentions': decoder_outputs['cross_attentions'],
                'encoder_last_hidden_state': encoder_outputs['last_hidden_state'],
                'encoder_hidden_states': encoder_outputs['hidden_states'],
                'encoder_attentions': encoder_outputs['attentions'],
            }

        return encoder_outputs

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.encoder.gradient_checkpointing_disable()
        self.decoder.gradient_checkpointing_disable()


class T5ForConditionalGeneration(nn.Module):
    """
    T5 Model for Conditional Generation (Text-to-Text)
    Adds a language modeling head for text generation
    """
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = config
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = config
        decoder_config.is_decoder = True 
        self.decoder = T5DecoderStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        """Initialize weights following T5 initialization scheme"""
        def _init_weights(module):
            """Initialize the weights"""
            factor = self.config.initializer_factor
            if isinstance(module, T5LayerNorm):
                module.weight.data.fill_(1.0)
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=factor * 1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()

        self.apply(_init_weights)

        # Tie word embeddings if specified
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.shared.weight

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        # Prepare decoder inputs
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Encode
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs['last_hidden_state']

        # Decode
        if decoder_input_ids is not None or decoder_inputs_embeds is not None:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            sequence_output = decoder_outputs['last_hidden_state']

            # Always apply scaling (remove conditional check)
            sequence_output = sequence_output * (self.model_dim ** -0.5)

            lm_logits = self.lm_head(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            return {
                'loss': loss,
                'logits': lm_logits,
                'past_key_values': decoder_outputs['past_key_values'],
                'decoder_hidden_states': decoder_outputs['hidden_states'],
                'decoder_attentions': decoder_outputs['attentions'],
                'cross_attentions': decoder_outputs['cross_attentions'],
                'encoder_last_hidden_state': encoder_outputs['last_hidden_state'],
                'encoder_hidden_states': encoder_outputs['hidden_states'],
                'encoder_attentions': encoder_outputs['attentions'],
            }

        return encoder_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _shift_right(self, input_ids):
        """Shift input ids one token to the right"""
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_length: int = 50,
        min_length: int = 1,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        early_stopping: bool = True,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        **kwargs
    ):
        """
        Simple greedy generation implementation
        For production use, consider implementing beam search and other advanced decoding strategies
        """
        # Properly define token IDs
        pad_token_id = self.config.pad_token_id
        eos_token_id = self.config.eos_token_id

        device = input_ids.device
        batch_size = input_ids.shape[0]  # Proper batch size extraction

        # Encode the input
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Initialize decoder input with decoder start token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device
        )

        # Generate tokens autoregressively
        for _ in range(max_length):
            # Forward pass through decoder
            outputs = self.forward(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            # Get logits for the last generated token
            next_token_logits = outputs['logits'][:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or take most likely token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

            # Check for early stopping
            if early_stopping and (next_token == eos_token_id).all():
                break

        return decoder_input_ids

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.encoder.gradient_checkpointing_disable()
        self.decoder.gradient_checkpointing_disable()


def create_t5_model(model_size: str = "small") -> T5ForConditionalGeneration:
    """
    Factory function to create T5 model of different sizes
    """
    size_configs = {
        "small": {
            "d_model": 512,
            "d_ff": 2048, 
            "num_layers": 6,
            "num_heads": 8,
        },
        "base": {
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12,
        },
        "large": {
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
        },
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Model size {model_size} not supported. Choose from {list(size_configs.keys())}")
    
    config = T5Config(**size_configs[model_size])
    return T5ForConditionalGeneration(config)
