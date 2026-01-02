"""
MemoryMAG: Titans-Style Neural Long-Term Memory for Transformers

This package implements a MAG (Memory as Gate) architecture that augments
transformers with neural long-term memory, enabling O(1) retrieval for
million-token contexts.

Main components:
- NeuralMemory: Deep MLP that learns to memorize at test time
- QueryProjector: Generates queries for memory retrieval with layer refinement
- MAGDecoderLayer: Augmented decoder layer with memory branch and gate
- patch_qwen3_with_mag: Utility to patch Qwen3 models with MAG
"""

from .neural_memory import NeuralMemory, NeuralMemoryWithDataDependentParams
from .query_projector import QueryProjector, MultiHeadQueryProjector
from .mag_layer import MAGDecoderLayer, MAGGate, MAGModelWrapper
from .patch_model import (
    Qwen3MAGConfig,
    Qwen3MAGDecoderLayer,
    Qwen3MAGModel,
    patch_qwen3_with_mag,
)
from .utils import (
    compute_surprise_score,
    adaptive_weight_decay,
    cosine_schedule,
    GateAnalyzer,
    MemoryDiagnostics,
    create_causal_mask,
    create_sliding_window_mask,
    count_parameters,
    get_parameter_groups,
)

__version__ = "0.1.0"
__all__ = [
    # Core components
    "NeuralMemory",
    "NeuralMemoryWithDataDependentParams",
    "QueryProjector",
    "MultiHeadQueryProjector",
    "MAGDecoderLayer",
    "MAGGate",
    "MAGModelWrapper",
    # Qwen3 patching
    "Qwen3MAGConfig",
    "Qwen3MAGDecoderLayer",
    "Qwen3MAGModel",
    "patch_qwen3_with_mag",
    # Utilities
    "compute_surprise_score",
    "adaptive_weight_decay",
    "cosine_schedule",
    "GateAnalyzer",
    "MemoryDiagnostics",
    "create_causal_mask",
    "create_sliding_window_mask",
    "count_parameters",
    "get_parameter_groups",
]
