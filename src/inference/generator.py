"""
T5 Inference and Generation Module
Advanced text generation with beam search, top-k, top-p sampling
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union
import numpy as np
from dataclasses import dataclass

from ..model.t5_model import T5ForConditionalGeneration
from ..data.tokenizer import T5Tokenizer


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 50
    min_length: int = 1
    num_beams: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = False
    early_stopping: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


class T5Generator:
    """
    Advanced T5 text generator with multiple decoding strategies
    """
    
    def __init__(
        self, 
        model: T5ForConditionalGeneration, 
        tokenizer: T5Tokenizer, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate(
        self,
        input_texts: Union[str, List[str]],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text using various decoding strategies
        """
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        
        # Tokenize inputs
        inputs = self.tokenizer.batch_encode(
            input_texts, 
            max_length=512, 
            padding=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            if generation_config.num_beams > 1:
                return self._beam_search_generate(
                    input_ids, attention_mask, generation_config
                )
            else:
                return self._greedy_or_sample_generate(
                    input_ids, attention_mask, generation_config
                )

    def _greedy_or_sample_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig
    ) -> List[str]:
        """Greedy decoding or sampling generation"""
        batch_size = input_ids.size(0)
        
        # Encode inputs
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Initialize decoder input
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.get_pad_token_id(),
            dtype=torch.long,
            device=self.device
        )
        
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(config.max_length):
            # Forward pass through decoder
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs['last_hidden_state'],
                encoder_attention_mask=attention_mask
            )
            
            # Get logits and apply scaling
            sequence_output = decoder_outputs['last_hidden_state']
            sequence_output = sequence_output * (self.model.model_dim ** -0.5)
            logits = self.model.lm_head(sequence_output)
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, decoder_input_ids, config.repetition_penalty
                )
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply top-k filtering
            if config.top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
            
            # Apply top-p filtering  
            if config.top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
            
            # Sample or select greedily
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
            
            # Check for EOS tokens
            finished = finished | (next_tokens.squeeze(-1) == self.tokenizer.get_eos_token_id())
            
            # Early stopping if all sequences finished
            if config.early_stopping and finished.all():
                break
        
        # Decode generated sequences
        return self._decode_sequences(decoder_input_ids)

    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig
    ) -> List[str]:
        """Beam search generation"""
        batch_size = input_ids.size(0)
        num_beams = config.num_beams
        
        # Encode inputs
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Expand for beam search
        encoder_hidden_states = encoder_outputs['last_hidden_state']
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(
            batch_size, num_beams, -1, -1
        ).reshape(batch_size * num_beams, -1, encoder_hidden_states.size(-1))
        
        encoder_attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, num_beams, -1
        ).reshape(batch_size * num_beams, -1)
        
        # Initialize beam search
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)
        
        # Initialize decoder input
        decoder_input_ids = torch.full(
            (batch_size * num_beams, 1),
            self.tokenizer.get_pad_token_id(),
            dtype=torch.long,
            device=self.device
        )
        
        for step in range(config.max_length):
            # Forward pass
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            
            # Get logits
            sequence_output = decoder_outputs['last_hidden_state']
            sequence_output = sequence_output * (self.model.model_dim ** -0.5)
            logits = self.model.lm_head(sequence_output)
            next_token_logits = logits[:, -1, :]
            
            # Apply length penalty
            if config.length_penalty != 1.0:
                length_penalty = ((5.0 + step + 1) / 6.0) ** config.length_penalty
                next_token_logits = next_token_logits / length_penalty
            
            # Get log probabilities
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            # Add beam scores
            next_token_scores = next_token_log_probs + beam_scores[:, None]
            
            # Reshape for beam search
            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Select top candidates
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # Determine beam and token indices
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Create new beams
            beam_outputs = []
            beam_next_tokens = []
            beam_idx = []
            
            for batch_idx in range(batch_size):
                beam_tokens = []
                beam_idxs = []
                beam_scores_batch = []
                
                for candidate_idx in range(2 * num_beams):
                    token_id = next_tokens[batch_idx, candidate_idx].item()
                    score = next_token_scores[batch_idx, candidate_idx].item()
                    beam_id = next_indices[batch_idx, candidate_idx].item()
                    
                    # Skip if EOS token and min length not reached
                    if (token_id == self.tokenizer.get_eos_token_id() and 
                        step + 1 < config.min_length):
                        continue
                    
                    beam_tokens.append(token_id)
                    beam_idxs.append(batch_idx * num_beams + beam_id)
                    beam_scores_batch.append(score)
                    
                    if len(beam_tokens) == num_beams:
                        break
                
                beam_outputs.extend(beam_scores_batch)
                beam_next_tokens.extend(beam_tokens)
                beam_idx.extend(beam_idxs)
            
            # Update beam scores and decoder input
            beam_scores = torch.tensor(beam_outputs, device=self.device)
            decoder_input_ids = decoder_input_ids[beam_idx]
            next_tokens_tensor = torch.tensor(beam_next_tokens, device=self.device).unsqueeze(-1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens_tensor], dim=-1)
            
            # Check for early stopping
            if config.early_stopping:
                # Implementation for early stopping in beam search
                pass
        
        # Select best beam for each batch
        decoder_input_ids = decoder_input_ids.view(batch_size, num_beams, -1)
        best_sequences = decoder_input_ids[:, 0, :]  # Take best beam
        
        return self._decode_sequences(best_sequences)

    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        input_ids: torch.Tensor, 
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        for i, input_seq in enumerate(input_ids):
            for token_id in set(input_seq.tolist()):
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty
        return logits

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits"""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits

    def _decode_sequences(self, sequences: torch.Tensor) -> List[str]:
        """Decode token sequences to text"""
        generated_texts = []
        
        for sequence in sequences:
            tokens = sequence.cpu().tolist()
            
            # Remove special tokens
            if self.tokenizer.get_eos_token_id() in tokens:
                tokens = tokens[:tokens.index(self.tokenizer.get_eos_token_id())]
            
            # Remove pad tokens
            tokens = [t for t in tokens if t != self.tokenizer.get_pad_token_id()]
            
            # Decode to text
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts

    def translate(self, text: str, source_lang: str, target_lang: str, **kwargs) -> str:
        """Convenience method for translation tasks"""
        task_text = f"translate {source_lang} to {target_lang}: {text}"
        results = self.generate([task_text], **kwargs)
        return results[0] if results else ""

    def summarize(self, text: str, **kwargs) -> str:
        """Convenience method for summarization tasks"""
        task_text = f"summarize: {text}"
        results = self.generate([task_text], **kwargs)
        return results[0] if results else ""

    def answer_question(self, question: str, context: str, **kwargs) -> str:
        """Convenience method for question answering tasks"""
        task_text = f"question: {question} context: {context}"
        results = self.generate([task_text], **kwargs)
        return results[0] if results else ""
