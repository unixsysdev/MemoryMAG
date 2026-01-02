# MemoryMAG - Project Overview

## Purpose
Implementation of a Titans-style MAG (Memory as a Gate) architecture that augments transformers with neural long-term memory. Enables O(1) retrieval complexity for million-token contexts.

## Core Concept
Add a parallel memory branch at every decoder layer:
- **Write**: Store information into fixed-size MLP weights based on "surprise metric"
- **Read**: Retrieve via learned query vectors that resonate with stored patterns
- **Gate**: Learned mixing between attention output and memory output

## Research Foundation
- **Titans Paper** (Google Research, 2024) - "Learning to Memorize at Test Time"
- **MIRAS Framework** - Theoretical unification of sequence modeling as associative memory
- Key insight: Deep MLP memory >> shallow matrix memory

## Architecture
```
For each decoder layer:
  Input → [Attention (frozen)] → attn_out
       → [QueryProjector] → [NeuralMemory] → ltm_out
       → [Gate] → h_out = residual + (1-g)*attn_out + g*ltm_out
       
  ltm_out passed to next layer's QueryProjector (query refinement)
```

## Base Model
- **Primary**: Qwen3-1.7B (local development)
- **Production**: Qwen3-4B (cloud training)

## Development Hardware
- Local: AMD Strix Halo, 128GB unified memory
- Production: A100/H100

## Training Strategy
1. Freeze base model weights
2. Train only: QueryProjectors, Gates, NMM initial states
3. Use BPTT-through-TTT (backprop through test-time training)
4. Curriculum: Hash-Hop → Dependency Tracing → Real Code

## Key Files
- `SPEC.md` - Full technical specification
- `src/neural_memory.py` - Deep MLP memory module
- `src/mag_layer.py` - Augmented decoder layer
- `src/patch_model.py` - Model patching utilities
- `data/generate_*.py` - Training data generation
- `training/train_mag.py` - Main training loop

## Current Status
- [x] Specification complete
- [ ] Architecture implementation
- [ ] Data generation
- [ ] Training loop
- [ ] Evaluation

## Critical Design Decisions
1. Memory is a 2-layer MLP, not a simple matrix
2. Surprise metric = reconstruction error with momentum
3. Query refinement: each layer's ltm_out feeds next layer's query
4. Adaptive weight decay prevents memory saturation
