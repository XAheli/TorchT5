"""
T5 Tokenizer Implementation
Wrapper around SentencePiece tokenizer used by T5 with proper text-to-text formatting
"""

import sentencepiece as spm
import os
import torch
from typing import List, Dict, Optional, Union


class T5Tokenizer:
    """
    T5 tokenizer wrapper around SentencePiece
    Handles text-to-text format with task prefixes and sentinel tokens
    """
    
    def __init__(self, model_file: str):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"SentencePiece model file not found: {model_file}")
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_file)
        
        # Special tokens
        self.eos_token = '</s>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.extra_ids = 100  # Number of sentinel tokens
        
        # Cache token IDs
        self._eos_token_id = self.sp_model.piece_to_id(self.eos_token)
        self._pad_token_id = self.sp_model.piece_to_id(self.pad_token)
        self._unk_token_id = self.sp_model.piece_to_id(self.unk_token)

    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """Encode text to token IDs"""
        token_ids = self.sp_model.encode(text, out_type=int)
        
        if max_length is not None:
            token_ids = token_ids[:max_length]
        
        if add_special_tokens:
            token_ids = token_ids + [self._eos_token_id]
            
        return token_ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if skip_special_tokens:
            # Remove special tokens
            ids = [id for id in ids if id not in [self._pad_token_id, self._eos_token_id]]
        
        return self.sp_model.decode(ids)

    def batch_encode(
        self, 
        texts: List[str], 
        padding: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Batch encode texts with padding"""
        all_token_ids = []
        
        for text in texts:
            token_ids = self.encode(text, max_length=max_length)
            all_token_ids.append(token_ids)
        
        if padding and max_length is not None:
            # Pad sequences to max_length
            for i, token_ids in enumerate(all_token_ids):
                pad_length = max_length - len(token_ids)
                if pad_length > 0:
                    all_token_ids[i] = token_ids + [self._pad_token_id] * pad_length
                elif pad_length < 0:
                    all_token_ids[i] = token_ids[:max_length]
        
        # Create attention mask
        attention_masks = []
        for token_ids in all_token_ids:
            attention_mask = [1 if id != self._pad_token_id else 0 for id in token_ids]
            attention_masks.append(attention_mask)
        
        if return_tensors == "pt":
            return {
                'input_ids': torch.tensor(all_token_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        
        return {
            'input_ids': all_token_ids,
            'attention_mask': attention_masks
        }

    def get_vocab_size(self) -> int:
        """Get total vocabulary size including sentinel tokens"""
        return self.sp_model.get_piece_size() + self.extra_ids

    def get_pad_token_id(self) -> int:
        return self._pad_token_id

    def get_eos_token_id(self) -> int:
        return self._eos_token_id

    def get_unk_token_id(self) -> int:
        return self._unk_token_id

    def get_sentinel_token_id(self, idx: int) -> int:
        """Get sentinel token ID for <extra_id_{idx}>"""
        if idx >= self.extra_ids:
            raise ValueError(f"Sentinel token index {idx} exceeds maximum {self.extra_ids}")
        return self.get_vocab_size() - self.extra_ids + idx

    def format_text_to_text(self, task: str, source: str, target: str = None) -> Dict[str, str]:
        """Format input for text-to-text training"""
        formatted_input = f"{task}: {source}"
        
        if target is not None:
            return {
                'input_text': formatted_input,
                'target_text': target
            }
        
        return {'input_text': formatted_input}

    def create_task_examples(self, task_type: str, examples: List[Dict]) -> List[Dict]:
        """Create text-to-text examples for different tasks"""
        formatted_examples = []
        
        for example in examples:
            if task_type == "translation":
                source_lang = example.get('source_lang', 'English')
                target_lang = example.get('target_lang', 'German')
                task_prefix = f"translate {source_lang} to {target_lang}"
                
            elif task_type == "summarization":
                task_prefix = "summarize"
                
            elif task_type == "question_answering":
                task_prefix = "question"
                source = f"{task_prefix}: {example['question']} context: {example['context']}"
                formatted_examples.append({
                    'input_text': source,
                    'target_text': example['answer']
                })
                continue
                
            elif task_type == "classification":
                task_prefix = f"classify sentiment"
                
            else:
                task_prefix = task_type
            
            formatted_example = self.format_text_to_text(
                task_prefix,
                example['source'],
                example.get('target')
            )
            formatted_examples.append(formatted_example)
        
        return formatted_examples


def download_t5_tokenizer(model_name: str = "t5-small") -> str:
    """Download T5 tokenizer model from Hugging Face"""
    try:
        from transformers import T5Tokenizer as HFT5Tokenizer
        
        # Download and save the SentencePiece model
        hf_tokenizer = HFT5Tokenizer.from_pretrained(model_name)
        model_file = f"{model_name}.model"
        
        # Save the SentencePiece model file
        with open(model_file, 'wb') as f:
            f.write(hf_tokenizer.sp_model.serialized_model_proto())
        
        return model_file
        
    except ImportError:
        raise ImportError("transformers library required to download tokenizer. Install with: pip install transformers")
