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

## Getting Started

See [SPEC.md](SPEC.md) for full technical specification.

### Requirements

```bash
pip install torch transformers einops tqdm
```

### Quick Start

```python
# Coming soon - architecture implementation in progress
from src.patch_model import patch_qwen3_with_mag

model = patch_qwen3_with_mag("Qwen/Qwen3-1.7B")
```

## Project Status

- [x] Technical specification
- [ ] Core architecture (NeuralMemory, MAGLayer)
- [ ] Model patching utilities
- [ ] Training data generation
- [ ] Training loop with BPTT-through-TTT
- [ ] Evaluation benchmarks

## Development

**Local Hardware**: AMD Strix Halo, 128GB unified memory  
**Base Model**: Qwen3-1.7B (development), Qwen3-4B (production)  
**Production Compute**: A100/H100

## Documentation

- [SPEC.md](SPEC.md) - Complete technical specification
- Architecture details, training curriculum, data generation

## References

1. Titans: Learning to Memorize at Test Time (Google Research, 2024)
2. MIRAS: A Unified Framework for Sequence Modeling (Google Research, 2024)
3. [Google Research Blog Post](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)

## License

MIT
