"""
Training Curriculum Management for MemoryMAG.

Implements the phased curriculum from the spec:
- Phase 1: Hash-Hop (exact match retrieval)
- Phase 2: Dependency Tracing (multi-hop reasoning)
- Phase 3: Real Code (semantic retrieval)

Each phase gradually increases memory pressure.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class CurriculumPhase(Enum):
    """Training curriculum phases."""
    HASH_HOP_SHORT = "1a"      # 8k-16k context
    HASH_HOP_LONG = "1b"       # 32k-64k context
    DEPENDENCY_2HOP = "2a"     # 2-hop reasoning
    DEPENDENCY_MULTI = "2b"    # 3-4 hop reasoning
    CODE_CLEAN = "3a"          # Clean repos
    CODE_MESSY = "3b"          # Complex repos


@dataclass
class PhaseConfig:
    """Configuration for a curriculum phase."""
    phase: CurriculumPhase
    data_path: str
    context_length: int
    num_steps: int
    learning_rate: float
    batch_size: int
    gradient_accumulation: int
    description: str
    
    # Success criteria
    target_accuracy: Optional[float] = None
    target_loss: Optional[float] = None


# Default curriculum phases
DEFAULT_CURRICULUM = [
    PhaseConfig(
        phase=CurriculumPhase.HASH_HOP_SHORT,
        data_path="data/hash_hop_16k.jsonl",
        context_length=16000,
        num_steps=1000,
        learning_rate=1e-4,
        batch_size=2,
        gradient_accumulation=4,
        description="Basic retrieval - gates learn to open",
        target_accuracy=0.95,
    ),
    PhaseConfig(
        phase=CurriculumPhase.HASH_HOP_LONG,
        data_path="data/hash_hop_64k.jsonl",
        context_length=64000,
        num_steps=2000,
        learning_rate=5e-5,
        batch_size=1,
        gradient_accumulation=8,
        description="Long-range retrieval - memory learns to persist",
        target_accuracy=0.90,
    ),
    PhaseConfig(
        phase=CurriculumPhase.DEPENDENCY_2HOP,
        data_path="data/dependency_2hop.jsonl",
        context_length=32000,
        num_steps=2000,
        learning_rate=5e-5,
        batch_size=1,
        gradient_accumulation=8,
        description="2-hop reasoning - query refinement basics",
        target_accuracy=0.85,
    ),
    PhaseConfig(
        phase=CurriculumPhase.DEPENDENCY_MULTI,
        data_path="data/dependency_4hop.jsonl",
        context_length=64000,
        num_steps=2000,
        learning_rate=3e-5,
        batch_size=1,
        gradient_accumulation=8,
        description="Multi-hop reasoning - deep query refinement",
        target_accuracy=0.80,
    ),
    PhaseConfig(
        phase=CurriculumPhase.CODE_CLEAN,
        data_path="data/code_clean.jsonl",
        context_length=64000,
        num_steps=5000,
        learning_rate=3e-5,
        batch_size=1,
        gradient_accumulation=8,
        description="Clean code repos - semantic retrieval",
        target_loss=2.0,
    ),
    PhaseConfig(
        phase=CurriculumPhase.CODE_MESSY,
        data_path="data/code_messy.jsonl",
        context_length=128000,
        num_steps=5000,
        learning_rate=1e-5,
        batch_size=1,
        gradient_accumulation=16,
        description="Complex repos - noise resistance",
        target_loss=2.5,
    ),
]


class CurriculumManager:
    """
    Manages the training curriculum.
    
    Handles:
    - Phase transitions
    - Data loading for each phase
    - Progress tracking
    - Early stopping criteria
    """
    
    def __init__(
        self,
        phases: Optional[List[PhaseConfig]] = None,
        checkpoint_dir: str = "checkpoints",
        start_phase: Optional[CurriculumPhase] = None,
    ):
        self.phases = phases or DEFAULT_CURRICULUM
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Find starting phase
        if start_phase:
            self.current_phase_idx = next(
                i for i, p in enumerate(self.phases) if p.phase == start_phase
            )
        else:
            self.current_phase_idx = 0
        
        # Progress tracking
        self.phase_history: List[Dict[str, Any]] = []
        self._load_history()
    
    @property
    def current_phase(self) -> PhaseConfig:
        """Get current phase configuration."""
        return self.phases[self.current_phase_idx]
    
    @property
    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.current_phase_idx >= len(self.phases)
    
    def advance_phase(self) -> bool:
        """
        Advance to next phase.
        
        Returns:
            True if advanced, False if curriculum complete
        """
        if self.current_phase_idx < len(self.phases) - 1:
            self.current_phase_idx += 1
            logger.info(f"Advanced to phase {self.current_phase.phase.value}: {self.current_phase.description}")
            return True
        return False
    
    def check_phase_complete(
        self,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
        step: int = 0,
    ) -> bool:
        """
        Check if current phase is complete.
        
        Phase is complete if:
        - Target accuracy/loss is reached, OR
        - Maximum steps is reached
        """
        phase = self.current_phase
        
        # Check step limit
        if step >= phase.num_steps:
            logger.info(f"Phase {phase.phase.value} complete: reached {step} steps")
            return True
        
        # Check accuracy target
        if phase.target_accuracy and accuracy and accuracy >= phase.target_accuracy:
            logger.info(f"Phase {phase.phase.value} complete: accuracy {accuracy:.2%} >= {phase.target_accuracy:.2%}")
            return True
        
        # Check loss target
        if phase.target_loss and loss and loss <= phase.target_loss:
            logger.info(f"Phase {phase.phase.value} complete: loss {loss:.4f} <= {phase.target_loss:.4f}")
            return True
        
        return False
    
    def record_phase_result(
        self,
        phase: CurriculumPhase,
        final_accuracy: Optional[float],
        final_loss: Optional[float],
        total_steps: int,
    ):
        """Record results from a completed phase."""
        result = {
            "phase": phase.value,
            "accuracy": final_accuracy,
            "loss": final_loss,
            "steps": total_steps,
        }
        self.phase_history.append(result)
        self._save_history()
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for current phase."""
        phase = self.current_phase
        return {
            "data_path": phase.data_path,
            "max_seq_length": phase.context_length // 4,  # Approximate tokens
            "learning_rate": phase.learning_rate,
            "batch_size": phase.batch_size,
            "gradient_accumulation_steps": phase.gradient_accumulation,
            "num_steps": phase.num_steps,
        }
    
    def _save_history(self):
        """Save phase history to checkpoint directory."""
        history_path = self.checkpoint_dir / "curriculum_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.phase_history, f, indent=2)
    
    def _load_history(self):
        """Load phase history from checkpoint directory."""
        history_path = self.checkpoint_dir / "curriculum_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.phase_history = json.load(f)
    
    def print_status(self):
        """Print curriculum status."""
        print("=" * 60)
        print("CURRICULUM STATUS")
        print("=" * 60)
        
        for i, phase in enumerate(self.phases):
            status = ""
            if i < self.current_phase_idx:
                # Find result in history
                result = next(
                    (r for r in self.phase_history if r["phase"] == phase.phase.value),
                    None
                )
                if result:
                    if result["accuracy"]:
                        status = f"✓ Complete (acc: {result['accuracy']:.2%})"
                    else:
                        status = f"✓ Complete (loss: {result['loss']:.4f})"
                else:
                    status = "✓ Complete"
            elif i == self.current_phase_idx:
                status = "→ Current"
            else:
                status = "  Pending"
            
            print(f"  Phase {phase.phase.value}: {phase.description}")
            print(f"    {status}")
            print(f"    Context: {phase.context_length:,} tokens, Steps: {phase.num_steps:,}")
        
        print("=" * 60)


def create_curriculum_from_config(config_path: str) -> CurriculumManager:
    """Create curriculum manager from JSON config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    phases = []
    for phase_config in config["phases"]:
        phases.append(PhaseConfig(
            phase=CurriculumPhase(phase_config["phase"]),
            data_path=phase_config["data_path"],
            context_length=phase_config["context_length"],
            num_steps=phase_config["num_steps"],
            learning_rate=phase_config["learning_rate"],
            batch_size=phase_config["batch_size"],
            gradient_accumulation=phase_config["gradient_accumulation"],
            description=phase_config.get("description", ""),
            target_accuracy=phase_config.get("target_accuracy"),
            target_loss=phase_config.get("target_loss"),
        ))
    
    return CurriculumManager(
        phases=phases,
        checkpoint_dir=config.get("checkpoint_dir", "checkpoints"),
    )


# Example curriculum config template
EXAMPLE_CONFIG = {
    "checkpoint_dir": "checkpoints",
    "phases": [
        {
            "phase": "1a",
            "data_path": "data/hash_hop_16k.jsonl",
            "context_length": 16000,
            "num_steps": 1000,
            "learning_rate": 1e-4,
            "batch_size": 2,
            "gradient_accumulation": 4,
            "description": "Basic retrieval",
            "target_accuracy": 0.95,
        },
        # ... more phases
    ]
}
