# T5 From Scratch: A Complete PyTorch Implementation

Work in progress!!  (filled with errors rn)

<!-- A comprehensive, production-ready implementation of the T5 (Text-to-Text Transfer Transformer) model from scratch in PyTorch, following the original research paper specifications.

## 🎯 Overview

This repository provides a complete implementation of T5 that includes:

- **Full T5 Architecture**: Encoder-decoder transformer with relative position encoding
- **Text-to-Text Framework**: Unified approach for all NLP tasks
- **Training Pipeline**: Pre-training and fine-tuning capabilities
- **Advanced Generation**: Beam search, top-k, top-p sampling
- **Production Ready**: Comprehensive testing, logging, and checkpointing

## 🏗️ Architecture

Our implementation follows the original T5 paper specifications:

- **Encoder-Decoder Architecture**: Bidirectional encoder + autoregressive decoder
- **Relative Position Encoding**: T5's unique position bias mechanism
- **Pre-Norm Layer Normalization**: Applied before attention and feed-forward layers
- **Text-to-Text Format**: All tasks framed as text generation

### Model Sizes Supported

| Model | d_model | d_ff | num_layers | num_heads | Parameters |
|-------|---------|------|------------|-----------|------------|
| Small | 512     | 2048 | 6          | 8         | ~60M       |
| Base  | 768     | 3072 | 12         | 12        | ~220M      |
| Large | 1024    | 4096 | 24         | 16        | ~770M      |


## 🎯 Features

### Model Features
- ✅ Full T5 architecture with relative position encoding
- ✅ Encoder-decoder attention mechanisms
- ✅ Text-to-text unified framework
- ✅ Multiple model sizes (Small, Base, Large)
- ✅ Gradient checkpointing for memory efficiency

### Training Features
- ✅ Pre-training with span corruption
- ✅ Fine-tuning for downstream tasks
- ✅ Mixed precision training support
- ✅ Learning rate scheduling with warmup
- ✅ Gradient clipping and accumulation
- ✅ Comprehensive logging and checkpointing

### Generation Features
- ✅ Greedy decoding
- ✅ Beam search
- ✅ Top-k and top-p sampling
- ✅ Temperature control
- ✅ Repetition penalty
- ✅ Length penalty

### Development Features
- ✅ Comprehensive test suite
- ✅ Type hints throughout
- ✅ Detailed documentation
- ✅ Example scripts and tutorials
- ✅ Configuration management

## 🔬 Research Compliance -->

This implementation strictly follows the original T5 paper:

> Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140), 1-67.

<!-- Key architectural choices:
- Pre-norm layer normalization placement
- Relative position bias computation
- Span corruption pre-training objective
- Text-to-text task formatting -->


