# MemoryMAG: Titans-Style Neural Long-Term Memory for Transformers

## Project Specification v1.0

---

## 1. Executive Summary

### 1.1 Project Overview

MemoryMAG implements a Titans-style MAG (Memory as a Gate) architecture that augments standard transformers with neural long-term memory. The core innovation is adding a parallel memory branch at every layer of the model, where information is written into fixed-size MLP weights during inference based on a "surprise metric" (how unexpected the input is), and retrieved via learned query vectors that each layer generates based on the current context.

This enables O(1) retrieval complexity regardless of context length, theoretically supporting million-token contexts without the quadratic scaling of standard attention.

### 1.2 Key Innovation

The model learns four critical skills through training:
1. **How to Save** - What information is worth remembering (via surprise metric)
2. **How to Write** - Encoding information in retrievable patterns (via fast-weight updates)
3. **How to Query** - Generating search vectors that resonate with stored patterns (via query projectors)
4. **When to Use Memory** - Deciding memory vs. local attention (via learned gates)

### 1.3 Research Foundation

This implementation is based on:
- **Titans Paper** (Google Research, 2024) - "Titans: Learning to Memorize at Test Time"
- **MIRAS Framework** (Google Research, 2024) - Theoretical unification of sequence modeling as associative memory

Key findings from the papers:
- Deep memory (multi-layer MLP) significantly outperforms shallow memory (single matrix)
- Momentum-based updates capture context around surprising tokens
- Adaptive weight decay prevents memory saturation over very long sequences
- The architecture scales to 2M+ token contexts while outperforming GPT-4 on long-context benchmarks

---

## 2. Technical Architecture

### 2.1 Base Model

**Target Model**: Qwen3-1.7B (primary), with scaling path to Qwen3-4B

**Rationale for 1.7B**:
- Large enough that retrieval success translates to task performance
- Small enough for rapid iteration on local hardware (Strix Halo, 128GB unified memory)
- Conclusions transfer directly to 4B with minimal code changes

**Development Hardware**:
- Local: AMD Strix Halo with 128GB unified memory
- Production: A100/H100 for full training runs

### 2.2 Neural Memory Module (NMM)

The memory is NOT a simple matrix - it is a **deep multi-layer perceptron** whose weights constitute the memory storage.

```
NeuralMemory:
    - Architecture: 2-layer MLP (d_model → d_hidden → d_model)
    - d_model: matches base model hidden dimension
    - d_hidden: typically 2x or 4x d_model
    - Initialization: Identity-like mapping with small Gaussian noise
    - Complexity: O(1) retrieval regardless of sequence length
```

**Write Operation (Surprise-Based)**:
```python
# Surprise Metric
L_surp = ||MLP(h_t; W_mem) - h_t||²

# Update Rule (Delta Rule with Momentum)
Δ_t = α * Δ_{t-1} - η * ∇L_surp
W_mem^{t} = W_mem^{t-1} + Δ_t

# Forgetting (Weight Decay)
W_mem^{t} = γ * W_mem^{t} where γ < 1
```

**Key Mechanisms**:
- **Momentum**: Captures tokens following a surprise event (even if individually unsurprising)
- **Weight Decay (γ)**: Adaptive forgetting to prevent memory saturation
- **Threshold**: Only update when surprise exceeds threshold to prevent "neural mush"

### 2.3 MAG Gateway (Per Layer)

Every decoder layer is augmented with a parallel memory branch:

```
┌─────────────────────────────────────────────────────────┐
│                    Decoder Layer N                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: h_in ──┬──────────────────┬─────────────────┐   │
│                │                  │                 │   │
│                ▼                  ▼                 │   │
│         ┌──────────┐      ┌─────────────┐          │   │
│         │  Self-   │      │   Query     │          │   │
│         │ Attention│      │ Projector   │          │   │
│         │ (frozen) │      │   (W_Q)     │          │   │
│         └────┬─────┘      └──────┬──────┘          │   │
│              │                   │                 │   │
│              │                   ▼                 │   │
│              │           ┌─────────────┐          │   │
│              │           │   Neural    │          │   │
│              │           │   Memory    │          │   │
│              │           │   (NMM)     │          │   │
│              │           └──────┬──────┘          │   │
│              │                  │                 │   │
│              ▼                  ▼                 │   │
│         attn_out            ltm_out               │   │
│              │                  │                 │   │
│              └────────┬─────────┘                 │   │
│                       │                           │   │
│                       ▼                           │   │
│              ┌─────────────┐                      │   │
│              │    Gate     │                      │   │
│              │  g = σ(W_g·h)                      │   │
│              └──────┬──────┘                      │   │
│                     │                             │   │
│                     ▼                             │   │
│    Output: h_out = h_in + (1-g)·attn_out + g·ltm_out │
│                                                   │   │
└─────────────────────────────────────────────────────┘

        │
        │ ltm_out passed to Layer N+1
        ▼
```

**Query Projector**:
```python
# Input: current hidden state + previous layer's memory output
q = W_Q @ concat(h_layer, r_prev_layer)

# This enables "query refinement" across layers:
# - Early layers: vague, broad queries
# - Middle layers: narrowing, focusing
# - Late layers: precise, exact retrieval
```

**Gate Logic**:
```python
# Learned gate determines memory vs attention mix
g = sigmoid(W_gate @ h_layer)

# Final output
h_out = h_residual + (1 - g) * attn_out + g * ltm_out
```

### 2.4 Information Flow Across Layers

The architecture enables "System 2" thinking through layered query refinement:

| Layer Range | Query Type | What It Retrieves |
|-------------|------------|-------------------|
| 0-20 (Early) | Syntax-focused | Token patterns, variable names, exact strings |
| 20-60 (Middle) | Relationship-focused | Dependencies, call chains, logical links |
| 60+ (Late) | Intent-focused | High-level patterns, coding style, goals |

Each layer's memory output (ltm_out) is passed to the next layer's QueryProjector, allowing progressive refinement of the search.

---

## 3. Training Specification

### 3.1 Training Philosophy: Meta-Learning

We are not just training next-token prediction - we are training the model to **manage its own memory**. This requires:

1. **BPTT-through-TTT**: Backpropagation Through Test-Time Training
   - The final prediction loss must flow back through the retrieval step
   - This teaches query projectors: "To reduce loss, you should have generated THIS query at layer 20"
   - **Stop-gradient fast updates**: Fast-weight updates are applied during the forward pass under `no_grad` for stability; gradients still flow through retrieval and gating

2. **Memory Pressure**: Data where local attention is insufficient
   - Forces gradients to flow through the LTM branch
   - Teaches the gate to open when memory is needed
   - Can be enforced by limiting attention to a fixed window during training (e.g., 4k)

### 3.2 Trainable vs Frozen Parameters

| Component | Status | Rationale |
|-----------|--------|-----------|
| Base Model (Attention, FFN, Embeddings) | **Frozen** | Preserve pre-trained intelligence |
| Query Projectors (W_Q at each layer) | **Trainable** | Learn "how to ask" |
| Gates (W_gate at each layer) | **Trainable** | Learn "when to use memory" |
| NMM Initial States | **Trainable** | Learn good starting point for memory |

**Estimated Trainable Parameters**: ~100-200M on top of frozen base model

### 3.3 Loss Function

```python
# Primary: Standard language modeling loss
L_LM = CrossEntropy(predictions, targets)

# Optional: Gate regularization (prevents collapse)
L_gate = gate_regularizer(gate_values)

# Total loss
L_total = L_LM + λ_gate * L_gate
```

### 3.4 Training Curriculum

Training proceeds in phases, gradually increasing memory pressure:

| Phase | Dataset | Context Length | Steps | Goal |
|-------|---------|----------------|-------|------|
| 1a | Hash-Hop (short) | 8k-16k | 1,000 | Gates learn to open |
| 1b | Hash-Hop (long) | 32k-64k | 2,000 | Memory learns to persist |
| 2a | Dependency Trace (2-hop) | 32k | 2,000 | Query refinement across layers |
| 2b | Dependency Trace (3-4 hop) | 64k | 2,000 | Deeper reasoning chains |
| 3a | Real Code (clean repos) | 64k-128k | 5,000 | Semantic retrieval |
| 3b | Real Code (messy repos) | 128k+ | 5,000 | Noise resistance |

---

## 4. Data Generation Specification

### 4.1 Phase 1: Synthetic Exact-Match (Hash-Hop)

**Purpose**: Validate that query projector can address specific memory locations.

```python
# Format
Context: "The secret key for {User_ID} is {random_hash}."
[... 50k-100k tokens of noise ...]
Query: "What is the secret key for {User_ID}?"
Target: "{random_hash}"

# Example
"CRITICAL DATA: The access token for User_a8f2k9 is x7z2m9q4p1."
[... massive haystack ...]
"Question: What is the access token for User_a8f2k9? Answer:"
→ "x7z2m9q4p1"
```

**Key Properties**:
- Random hashes are incompressible - can't be guessed
- Forces model to actually use memory (attention window too small)
- Randomize needle position to prevent positional shortcuts

### 4.2 Phase 2: Synthetic Reasoning (Dependency Tracing)

**Purpose**: Teach multi-hop retrieval and query refinement.

```python
# Format (2-hop)
Fact 1: "Module {A} depends on Module {B}."
[... 30k tokens ...]
Fact 2: "Module {B} requires {C} to function."
[... 30k tokens ...]
Query: "Does Module {A} ultimately require {C}?"
Target: "Yes"

# Format (3-hop)
Fact 1: "Agent Alpha supervises Agent Beta."
[... noise ...]
Fact 2: "Agent Beta revoked access for Agent Gamma."
[... noise ...]
Fact 3: "Agent Gamma was trying to open Vault 7."
[... noise ...]
Query: "Who is responsible for the security breach at Vault 7?"
Target: "Agent Alpha"
```

**Key Properties**:
- Can't be solved by finding single needle
- Requires combining multiple stored facts
- Tests layer-to-layer query refinement

### 4.3 Phase 3: Real Code Repositories

**Purpose**: Generalize to actual software development tasks.

**Data Extraction Strategy**:

```python
def generate_code_retrieval_sample(repo_path):
    # 1. Parse repo, build dependency graph (AST-based)
    # 2. Find function that uses something defined far away
    # 3. Create context: [definition] + [noise files] + [usage site]
    # 4. Create task: complete the usage site
    # 5. Target: completion requiring the distant definition
```

**Example**:
```python
# Token ~500 (utils/config.py)
DATABASE_TIMEOUT = 30
RETRY_COUNT = 3

# [... 80k tokens of other files ...]

# Token ~80000 (services/db_handler.py)
def connect_with_retry():
    for i in range(???):  # Model must retrieve RETRY_COUNT
        conn = db.connect(timeout=???)  # Must retrieve DATABASE_TIMEOUT
```

**Repository Curriculum**:
1. **Small, Clean**: requests, flask - consistent style, clear dependencies
2. **Messy, Polyglot**: Full-stack React + Node - mixed syntax, complex structure
3. **Massive Monorepos**: Large-scale repos requiring 1M+ token retrieval

### 4.4 Data Generation Scripts Required

| Script | Output | Samples |
|--------|--------|---------|
| `generate_hash_hop.py` | `data/hash_hop_{context_len}.jsonl` | 10k-50k |
| `generate_dependency.py` | `data/dependency_{hops}hop.jsonl` | 10k-50k |
| `generate_code_retrieval.py` | `data/code_{repo_type}.jsonl` | 10k+ |

---

## 5. Implementation Specification

### 5.1 Required Files

```
MemoryMAG/
├── SPEC.md                    # This document
├── README.md                  # Quick start guide
│
├── src/
│   ├── __init__.py
│   ├── neural_memory.py       # NeuralMemory class (the deep MLP memory)
│   ├── mag_layer.py           # MAGDecoderLayer (augmented decoder)
│   ├── query_projector.py     # QueryProjector with layer refinement
│   ├── patch_model.py         # Monkey-patch Qwen3 with MAG layers
│   └── utils.py               # Surprise metric, weight decay, etc.
│
├── data/
│   ├── generate_hash_hop.py
│   ├── generate_dependency.py
│   ├── generate_code_retrieval.py
│   └── *.jsonl                # Generated datasets
│
├── training/
│   ├── train_mag.py           # Main training loop
│   ├── curriculum.py          # Training phase management
│   └── diagnostics.py         # Gate activation monitoring
│
└── evaluation/
    ├── needle_test.py         # Needle-in-haystack benchmarks
    └── code_completion.py     # Real-world code tasks
```

### 5.2 Core Class Interfaces

**NeuralMemory**:
```python
class NeuralMemory(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        """Initialize 2-layer MLP memory."""
        
    def retrieve(self, query: Tensor) -> Tensor:
        """O(1) retrieval via forward pass through MLP."""
        
    def compute_surprise(self, hidden_state: Tensor) -> Tensor:
        """Compute reconstruction error as surprise metric."""
        
    def fast_update(self, hidden_state: Tensor, surprise: Tensor):
        """Apply delta rule with momentum if surprise > threshold."""
        
    def apply_decay(self):
        """Apply weight decay (forgetting factor)."""
```

**MAGDecoderLayer**:
```python
class MAGDecoderLayer(nn.Module):
    def __init__(self, original_layer, d_model: int, ...):
        """Wrap original decoder layer with MAG components."""
        self.original_layer = original_layer  # Frozen
        self.neural_memory = NeuralMemory(d_model, d_hidden)
        self.query_projector = QueryProjector(d_model)
        self.gate = nn.Linear(d_model, 1)
        
    def forward(self, hidden_states, prev_ltm_out=None, ...):
        """
        Dual-path forward:
        1. Original attention (frozen)
        2. Memory retrieval (trainable)
        3. Gated fusion
        """
```

**QueryProjector**:
```python
class QueryProjector(nn.Module):
    def __init__(self, d_model: int):
        """Projects hidden state + prev memory output to query vector."""
        
    def forward(self, hidden_state: Tensor, prev_ltm_out: Tensor = None) -> Tensor:
        """Generate query for memory lookup."""
```

### 5.3 Model Patching Strategy

```python
def patch_qwen3_with_mag(model_name: str = "Qwen/Qwen3-1.7B"):
    """
    Load pre-trained Qwen3 and surgically inject MAG layers.
    
    1. Load original model
    2. For each decoder layer:
       - Wrap in MAGDecoderLayer
       - Freeze original weights
       - Initialize trainable MAG components
    3. Modify forward pass to propagate ltm_out between layers
    """
```

---

## 6. Diagnostics & Success Metrics

### 6.1 Training Diagnostics

| Metric | Expected Behavior | Red Flag |
|--------|-------------------|----------|
| Gate Activation (g) | Start ~0.05, grow to ~0.80 on memory-required tokens | Stuck at 0 or 1 |
| Surprise Distribution | Spike on needles, low on noise | Uniform (everything surprising) |
| Query-Memory Cosine Sim | Increase during training | Random/unchanging |
| Layer-wise Gate Pattern | Early layers low, late layers high on retrieval tasks | Uniform across layers |

### 6.2 Evaluation Benchmarks

| Benchmark | Target | Description |
|-----------|--------|-------------|
| Hash-Hop @ 64k | 100% | Exact-match retrieval |
| Hash-Hop @ 128k | >95% | Long-range retrieval |
| Dependency Trace (3-hop) | >90% | Multi-step reasoning |
| BABILong | Outperform baseline Qwen3 | Standard long-context benchmark |
| Code Completion | Qualitative | Real-world utility |

---

## 7. Development Phases

### Phase 1: Architecture Validation (Local, 1.7B)
- [ ] Implement NeuralMemory class
- [ ] Implement MAGDecoderLayer
- [ ] Implement model patching
- [ ] Verify forward pass works
- [ ] Verify gradients flow through MAG components
- [ ] Generate Phase 1 synthetic data

### Phase 2: Training Validation (Local, 1.7B)
- [ ] Implement training loop with BPTT-through-TTT
- [ ] Run Hash-Hop curriculum
- [ ] Monitor gate activations
- [ ] Validate retrieval on synthetic data
- [ ] Generate Phase 2 & 3 data

### Phase 3: Full Training (Cloud, 4B)
- [ ] Scale to Qwen3-4B
- [ ] Run full curriculum
- [ ] Evaluate on BABILong
- [ ] Real code completion tests

---

## 8. Open Questions & Risks

### 8.1 Technical Uncertainties

1. **Memory Depth**: Paper says deep > shallow, but optimal depth for 1.7B/4B unknown
2. **Surprise Threshold**: How to calibrate? May need adaptive thresholding
3. **Forgetting Rate**: γ value for weight decay - too high = memory loss, too low = saturation
4. **Query Projector Input**: Concat vs. add for combining h_layer and prev_ltm_out?

### 8.2 Risks

1. **Synthetic → Real Generalization**: Hash-Hop may not transfer to semantic retrieval
2. **Training Instability**: BPTT-through-TTT is notoriously finicky
3. **Compute Requirements**: May need more steps than estimated for convergence
4. **Gate Collapse**: Gate may learn to stay closed (ignore memory) or always open (ignore attention)

### 8.3 Mitigation Strategies

- Start with simple architecture, add complexity only if needed
- Extensive logging of gate activations at every layer
- Curriculum designed to force memory usage
- Regularization to prevent gate collapse

---

## 9. References

1. **Titans Paper**: "Titans: Learning to Memorize at Test Time" - Google Research, 2024
2. **MIRAS Paper**: "MIRAS: A Unified Framework for Sequence Modeling" - Google Research, 2024
3. **Google Research Blog**: "Titans + MIRAS: Helping AI have long-term memory" - December 2025
4. **Previous Work**: MAC (Memory as Context) experiments - showed identity mapping insufficient

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Status: Ready for Implementation*
