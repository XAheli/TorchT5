"""
T5 Dataset Implementation
Handles text-to-text data loading and preprocessing for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union
import json
import random
from .tokenizer import T5Tokenizer


class T5TextDataset(Dataset):
    """
    Dataset for T5 text-to-text training
    Supports various NLP tasks in unified text-to-text format
    """
    
    def __init__(
        self, 
        examples: List[Dict[str, str]], 
        tokenizer: T5Tokenizer, 
        max_source_length: int = 512, 
        max_target_length: int = 50,
        task_type: str = "general"
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Extract source and target text
        source_text = example['input_text']
        target_text = example.get('target_text', '')
        
        # Tokenize source
        source_encoding = self.tokenizer.batch_encode(
            [source_text],
            padding=True,
            max_length=self.max_source_length,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_ids = self.tokenizer.encode(
            target_text, 
            max_length=self.max_target_length
        )
        
        # Pad target
        target_ids = target_ids + [self.tokenizer.get_pad_token_id()] * (
            self.max_target_length - len(target_ids)
        )
        target_ids = target_ids[:self.max_target_length]
        
        # Create labels (replace padding tokens with -100 for loss computation)
        labels = target_ids.copy()
        labels = [-100 if token == self.tokenizer.get_pad_token_id() else token for token in labels]
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class T5PretrainingDataset(Dataset):
    """
    Dataset for T5 pre-training with span corruption task
    Implements the span masking approach from the original T5 paper
    """
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: T5Tokenizer,
        max_length: int = 512,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Truncate if too long
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # Apply span corruption
        corrupted_input, target = self._apply_span_corruption(token_ids)
        
        # Encode corrupted input and target
        input_encoding = self.tokenizer.batch_encode(
            [corrupted_input],
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        target_ids = self.tokenizer.encode(target, max_length=self.max_length)
        target_ids = target_ids + [self.tokenizer.get_pad_token_id()] * (
            self.max_length - len(target_ids)
        )
        target_ids = target_ids[:self.max_length]
        
        # Create labels
        labels = [-100 if token == self.tokenizer.get_pad_token_id() else token for token in target_ids]
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def _apply_span_corruption(self, token_ids: List[int]) -> Tuple[str, str]:
        """Apply span corruption masking as in T5 pre-training"""
        # Calculate number of tokens to mask
        num_tokens_to_mask = int(len(token_ids) * self.noise_density)
        
        if num_tokens_to_mask == 0:
            # Return original if no masking needed
            original_text = self.tokenizer.decode(token_ids)
            return original_text, ""
        
        # Create spans to mask
        mask_spans = self._create_mask_spans(len(token_ids), num_tokens_to_mask)
        
        # Apply masking and create target
        corrupted_tokens = []
        target_tokens = []
        sentinel_id = 0
        
        i = 0
        while i < len(token_ids):
            if any(start <= i < end for start, end in mask_spans):
                # Find the span this token belongs to
                span_start, span_end = next((start, end) for start, end in mask_spans if start <= i < end)
                
                # Add sentinel token to corrupted input
                sentinel_token = f"<extra_id_{sentinel_id}>"
                corrupted_tokens.append(sentinel_token)
                
                # Add sentinel and original tokens to target
                target_tokens.append(sentinel_token)
                target_tokens.extend([self.tokenizer.decode([token_ids[j]]) for j in range(span_start, span_end)])
                
                sentinel_id += 1
                i = span_end
            else:
                corrupted_tokens.append(self.tokenizer.decode([token_ids[i]]))
                i += 1
        
        corrupted_text = " ".join(corrupted_tokens)
        target_text = " ".join(target_tokens)
        
        return corrupted_text, target_text

    def _create_mask_spans(self, length: int, num_tokens_to_mask: int) -> List[Tuple[int, int]]:
        """Create random spans to mask"""
        # Calculate number of spans
        num_spans = max(1, round(num_tokens_to_mask / self.mean_noise_span_length))
        
        # Create spans
        spans = []
        masked_tokens = 0
        
        for _ in range(num_spans):
            if masked_tokens >= num_tokens_to_mask:
                break
            
            # Random span length around mean
            span_length = max(1, int(random.gauss(self.mean_noise_span_length, 1)))
            span_length = min(span_length, num_tokens_to_mask - masked_tokens)
            
            # Random start position
            max_start = length - span_length
            if max_start <= 0:
                break
                
            start = random.randint(0, max_start)
            end = start + span_length
            
            # Check for overlap with existing spans
            if not any(start < s_end and end > s_start for s_start, s_end in spans):
                spans.append((start, end))
                masked_tokens += span_length
        
        return sorted(spans)


def load_dataset_from_json(file_path: str) -> List[Dict[str, str]]:
    """Load dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for training"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for T5 batches"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
