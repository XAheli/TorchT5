"""
T5 Model Configuration
Following the original T5 paper specifications with text-to-text framework
"""

import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class T5Config:
    """Configuration class for T5 model following original paper specifications"""
    
    # Model architecture
    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    
    # Attention configuration  
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    
    # Regularization
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    
    # Initialization
    initializer_factor: float = 1.0
    
    # Architecture specifics
    feed_forward_proj: str = "relu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    
    # Special tokens
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    
    # Additional parameters
    tie_word_embeddings: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str, model_name: str = "t5_small") -> "T5Config":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_config = config_dict.get(model_name, {})
        return cls(**model_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        
        if self.d_kv * self.num_heads != self.d_model:
            # Auto-adjust d_kv if not explicitly set correctly
            self.d_kv = self.d_model // self.num_heads
