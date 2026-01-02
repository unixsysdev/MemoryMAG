# MemoryMAG

**Titans-Style Neural Long-Term Memory for Transformers**

## Overview

MemoryMAG implements a MAG (Memory as a Gate) architecture based on Google's Titans paper. It augments standard transformers with neural long-term memory, enabling O(1) retrieval complexity for million-token contexts.

### Key Innovation

Instead of relying solely on attention (which scales O(n²) with context), we add a parallel memory branch at every layer:

1. **Write**: Information stored into fixed-size MLP weights based on "surprise" (how unexpected the input is)
2. **Read**: Retrieved via learned query vectors that resonate with stored patterns
3. **Gate**: Learned mixing between attention and memory outputs

The model learns *how to save*, *how to query*, and *when to use memory vs. attention* - all through backpropagation.

## Architecture

```
┌─────────────────────────────────────────┐
│           Decoder Layer N               │
├─────────────────────────────────────────┤
│                                         │
│  h_in ──┬── [Attention] ──→ attn_out    │
│         │                      │        │
│         └── [Query Proj] ──→ [NMM] ──→ ltm_out
│                                │        │
│              ┌─────────────────┘        │
│              ▼                          │
│         [Gate: g = σ(W·h)]              │
│              │                          │
│              ▼                          │
│  h_out = residual + (1-g)·attn + g·ltm  │
└─────────────────────────────────────────┘
```

## Research Foundation

- **Titans Paper**: "Learning to Memorize at Test Time" (Google Research, 2024)
- **MIRAS Framework**: Theoretical unification of sequence modeling as associative memory
- Results: Outperforms GPT-4 on long-context benchmarks, scales to 2M+ tokens

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers einops tqdm accelerate

# For Qwen3 models
pip install transformers>=4.40.0
```

## Quick Start

### 1. Patch a Model with MAG

```python
from src import patch_qwen3_with_mag, Qwen3MAGConfig

# Configure MAG components
config = Qwen3MAGConfig(
    memory_layers=2,           # Depth of memory MLP
    n_persistent_tokens=16,    # Learned prefix tokens
    chunk_size=64,             # Memory update chunk size
)

# Load and patch model
model = patch_qwen3_with_mag(
    model_name_or_path="Qwen/Qwen3-1.7B",
    config=config,
    device="auto",
    dtype=torch.bfloat16,
)

# Check trainable parameters
trainable = model.count_trainable_parameters()
print(f"Trainable: {trainable:,} parameters")
```

### 2. Generate Training Data

```bash
# Phase 1: Hash-Hop (exact match retrieval)
python data/generate_hash_hop.py \
    --context_len 64000 \
    --num_samples 10000 \
    --output data/hash_hop_64k.jsonl

# Phase 2: Dependency Tracing (multi-hop reasoning)
python data/generate_dependency.py \
    --hops 3 \
    --num_samples 10000 \
    --output data/dependency_3hop.jsonl

# Phase 3: Code Retrieval (synthetic)
python data/generate_code_retrieval.py \
    --synthetic \
    --num_samples 10000 \
    --output data/code_retrieval.jsonl
```

### 3. Train MAG Components

```bash
# Phase 1: Hash-Hop (exact retrieval)
python training/train_mag.py \
    --model_name Qwen/Qwen3-1.7B \
    --data_path data/hash_hop_16k.jsonl \
    --output_dir checkpoints/phase1 \
    --learning_rate 1e-4 \
    --num_epochs 3

# Phase 2: Dependency (builds on Phase 1 checkpoint)
python training/train_mag.py \
    --model_name Qwen/Qwen3-1.7B \
    --data_path data/dependency_2hop.jsonl \
    --output_dir checkpoints/phase2 \
    --resume_from checkpoints/phase1/best \
    --learning_rate 5e-5 \
    --num_epochs 3

# Phase 3: Code (builds on Phase 2 checkpoint)
python training/train_mag.py \
    --model_name Qwen/Qwen3-1.7B \
    --data_path data/code_retrieval.jsonl \
    --output_dir checkpoints/phase3 \
    --resume_from checkpoints/phase2/best \
    --learning_rate 3e-5 \
    --num_epochs 3
```

Checkpoints chain automatically - each phase builds on the previous phase's learned weights.

### 4. Evaluate

```bash
# Needle-in-haystack benchmark
python evaluation/needle_test.py \
    --checkpoint checkpoints/best \
    --context_lengths 2000,4000,8000,16000

# Code completion evaluation
python evaluation/code_completion.py \
    --checkpoint checkpoints/best
```

### 5. Monitor Training

```bash
# Run diagnostics on checkpoint
python training/diagnostics.py \
    --checkpoint checkpoints/latest
```

## Project Structure

```
MemoryMAG/
├── SPEC.md                    # Full technical specification
├── README.md                  # This file
│
├── src/
│   ├── __init__.py
│   ├── neural_memory.py       # Deep MLP memory module
│   ├── query_projector.py     # Query generation with layer refinement
│   ├── mag_layer.py           # Augmented decoder layer
│   ├── patch_model.py         # Qwen3 patching utilities
│   └── utils.py               # Helper functions
│
├── data/
│   ├── generate_hash_hop.py   # Phase 1 data
│   ├── generate_dependency.py # Phase 2 data
│   └── generate_code_retrieval.py  # Phase 3 data
│
├── training/
│   ├── train_mag.py           # Main training loop
│   ├── curriculum.py          # Training phase management
│   └── diagnostics.py         # Gate/memory monitoring
│
└── evaluation/
    ├── needle_test.py         # Needle-in-haystack
    └── code_completion.py     # Code completion tasks
```

## Training Curriculum

Training proceeds in phases with increasing memory pressure:

| Phase | Data | Context | Goal |
|-------|------|---------|------|
| 1a | Hash-Hop (short) | 8k-16k | Gates learn to open |
| 1b | Hash-Hop (long) | 32k-64k | Memory learns to persist |
| 2a | Dependency (2-hop) | 32k | Query refinement basics |
| 2b | Dependency (3-4 hop) | 64k | Deep reasoning chains |
| 3a | Code (clean) | 64k-128k | Semantic retrieval |
| 3b | Code (complex) | 128k+ | Noise resistance |

## Key Components

### Neural Memory Module

The memory is a 2+ layer MLP whose weights constitute the memory storage:

- **Surprise metric**: Reconstruction error drives updates
- **Momentum**: Captures context around surprising events
- **Weight decay**: Adaptive forgetting prevents saturation

### Query Refinement

Each layer's memory output feeds the next layer's query projector:

- Early layers: Broad, syntax-focused queries
- Middle layers: Relationship-focused queries
- Late layers: Intent-focused, precise queries

### Gate

Learns when to use attention vs. memory:

- Starts biased toward attention (gate ≈ 0)
- Opens on tokens requiring long-range retrieval
- Per-layer, per-token decisions

## Hardware Requirements

- **Development**: 128GB+ RAM, GPU with 24GB+ VRAM
- **Target**: Qwen3-1.7B fits on consumer GPUs
- **Production**: A100/H100 for full training

## Documentation

- [SPEC.md](SPEC.md) - Complete technical specification
- Detailed architecture, training strategies, and success metrics

## References

1. Titans: Learning to Memorize at Test Time (Google Research, 2024)
2. MIRAS: A Unified Framework for Sequence Modeling (Google Research, 2024)
3. [Google Research Blog](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)

## License

MIT
