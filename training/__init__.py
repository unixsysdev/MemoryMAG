"""
MemoryMAG Training Module.

Contains:
- train_mag.py: Main training script with BPTT-through-TTT
- curriculum.py: Training phase management
- diagnostics.py: Gate and memory monitoring
"""

from .curriculum import (
    CurriculumPhase,
    PhaseConfig,
    CurriculumManager,
    DEFAULT_CURRICULUM,
)
