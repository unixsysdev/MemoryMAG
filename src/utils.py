"""
Utility functions for MemoryMAG.

Common utilities for:
- Surprise metric computation
- Weight decay scheduling
- Gate activation analysis
- Memory diagnostics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


def compute_surprise_score(
    memory_output: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'none'
) -> torch.Tensor:
    """
    Compute surprise as reconstruction error.
    
    Surprise = ||Memory(key) - value||²
    
    Args:
        memory_output: Output from memory forward pass
        target: Target values (what we expected)
        reduction: 'none', 'mean', 'sum'
        
    Returns:
        Surprise scores
    """
    surprise = F.mse_loss(memory_output, target, reduction=reduction)
    if reduction == 'none':
        # Sum over feature dimension to get per-token surprise
        surprise = surprise.sum(dim=-1)
    return surprise


def adaptive_weight_decay(
    current_surprise: torch.Tensor,
    max_surprise: float = 10.0,
    min_decay: float = 0.001,
    max_decay: float = 0.1,
) -> torch.Tensor:
    """
    Compute adaptive weight decay based on surprise level.
    
    Higher surprise → lower decay (keep memory longer)
    Lower surprise → higher decay (forget faster)
    
    Args:
        current_surprise: Current surprise value
        max_surprise: Surprise value that maps to min_decay
        min_decay: Minimum weight decay
        max_decay: Maximum weight decay
        
    Returns:
        Weight decay value
    """
    # Normalize surprise to [0, 1]
    normalized = torch.clamp(current_surprise / max_surprise, 0, 1)
    
    # Inverse relationship: high surprise → low decay
    decay = max_decay - normalized * (max_decay - min_decay)
    
    return decay


def cosine_schedule(
    step: int,
    total_steps: int,
    start_value: float,
    end_value: float,
    warmup_steps: int = 0,
) -> float:
    """
    Cosine annealing schedule with warmup.
    
    Args:
        step: Current step
        total_steps: Total number of steps
        start_value: Starting value
        end_value: Ending value
        warmup_steps: Number of warmup steps
        
    Returns:
        Scheduled value
    """
    if step < warmup_steps:
        # Linear warmup
        return start_value * step / warmup_steps
    
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_value = 0.5 * (1 + math.cos(math.pi * progress))
    
    return end_value + (start_value - end_value) * cosine_value


class GateAnalyzer:
    """
    Analyze gate activations during training/inference.
    
    Tracks:
    - Mean gate values per layer
    - Gate values on "needle" vs "haystack" tokens
    - Gate activation patterns over training
    """
    
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.history: Dict[str, List] = {
            'mean_gates': [],
            'layer_gates': [[] for _ in range(n_layers)],
            'steps': [],
        }
        self.current_step = 0
    
    def record(
        self,
        gate_values: List[torch.Tensor],
        step: Optional[int] = None,
    ):
        """
        Record gate values from all layers.
        
        Args:
            gate_values: List of gate tensors [batch, seq, d_model] per layer
            step: Training step (auto-increments if None)
        """
        if step is None:
            step = self.current_step
            self.current_step += 1
        
        self.history['steps'].append(step)
        
        mean_gates = []
        for layer_idx, gates in enumerate(gate_values):
            if gates is not None:
                mean_gate = gates.mean().item()
                self.history['layer_gates'][layer_idx].append(mean_gate)
                mean_gates.append(mean_gate)
        
        self.history['mean_gates'].append(sum(mean_gates) / len(mean_gates) if mean_gates else 0)
    
    def get_layer_statistics(self, layer_idx: int) -> Dict[str, float]:
        """Get statistics for a specific layer."""
        gates = self.history['layer_gates'][layer_idx]
        if not gates:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'mean': sum(gates) / len(gates),
            'std': (sum((g - sum(gates)/len(gates))**2 for g in gates) / len(gates)) ** 0.5,
            'min': min(gates),
            'max': max(gates),
        }
    
    def detect_collapse(self, threshold: float = 0.05) -> List[int]:
        """
        Detect layers where gate might have collapsed (stuck at 0 or 1).
        
        Returns:
            List of layer indices with potential collapse
        """
        collapsed = []
        for layer_idx in range(self.n_layers):
            stats = self.get_layer_statistics(layer_idx)
            if stats['std'] < threshold and (stats['mean'] < 0.1 or stats['mean'] > 0.9):
                collapsed.append(layer_idx)
        return collapsed
    
    def print_summary(self):
        """Print analysis summary."""
        print("=" * 50)
        print("Gate Activation Analysis")
        print("=" * 50)
        
        for layer_idx in range(self.n_layers):
            stats = self.get_layer_statistics(layer_idx)
            print(f"Layer {layer_idx:2d}: mean={stats['mean']:.3f} std={stats['std']:.3f} "
                  f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        collapsed = self.detect_collapse()
        if collapsed:
            print(f"\n⚠️  Warning: Potential gate collapse in layers: {collapsed}")
        
        print("=" * 50)


class MemoryDiagnostics:
    """
    Diagnostics for neural memory during training.
    
    Tracks:
    - Surprise distribution
    - Memory weight norms
    - Gradient norms
    - Query-memory alignment
    """
    
    def __init__(self):
        self.history: Dict[str, List] = {
            'surprise_mean': [],
            'surprise_std': [],
            'weight_norms': [],
            'gradient_norms': [],
            'alignment_scores': [],
        }
    
    def record_surprise(self, surprise: torch.Tensor):
        """Record surprise statistics."""
        self.history['surprise_mean'].append(surprise.mean().item())
        self.history['surprise_std'].append(surprise.std().item())
    
    def record_weight_norm(self, weights: List[torch.Tensor]):
        """Record memory weight norms."""
        total_norm = sum(w.norm().item() for w in weights)
        self.history['weight_norms'].append(total_norm)
    
    def record_gradient_norm(self, gradients: List[torch.Tensor]):
        """Record gradient norms."""
        total_norm = sum(g.norm().item() for g in gradients if g is not None)
        self.history['gradient_norms'].append(total_norm)
    
    def compute_alignment(
        self,
        query: torch.Tensor,
        memory_output: torch.Tensor,
    ) -> float:
        """
        Compute query-memory alignment (cosine similarity).
        
        Higher alignment = query successfully retrieves relevant info.
        """
        # Normalize
        q_norm = F.normalize(query.flatten(), dim=0)
        m_norm = F.normalize(memory_output.flatten(), dim=0)
        
        alignment = (q_norm * m_norm).sum().item()
        self.history['alignment_scores'].append(alignment)
        
        return alignment
    
    def check_memory_health(self) -> Dict[str, str]:
        """
        Check memory health indicators.
        
        Returns:
            Dict with health status for each indicator
        """
        health = {}
        
        # Check surprise - should have reasonable variance
        if self.history['surprise_mean']:
            recent_surprises = self.history['surprise_mean'][-100:]
            if all(s < 0.01 for s in recent_surprises):
                health['surprise'] = '⚠️  Low (memory might be saturated)'
            elif all(s > 10 for s in recent_surprises):
                health['surprise'] = '⚠️  High (memory not learning)'
            else:
                health['surprise'] = '✓ Healthy'
        
        # Check weight norms - shouldn't explode or vanish
        if self.history['weight_norms']:
            recent_norms = self.history['weight_norms'][-100:]
            if max(recent_norms) > 100 * min(recent_norms):
                health['weights'] = '⚠️  Unstable (large variance)'
            elif max(recent_norms) > 1000:
                health['weights'] = '⚠️  Large (might need more decay)'
            else:
                health['weights'] = '✓ Healthy'
        
        # Check alignment - should improve over training
        if len(self.history['alignment_scores']) > 100:
            early = sum(self.history['alignment_scores'][:50]) / 50
            late = sum(self.history['alignment_scores'][-50:]) / 50
            if late < early:
                health['alignment'] = '⚠️  Decreasing (query learning issue)'
            else:
                health['alignment'] = '✓ Improving'
        
        return health


def create_causal_mask(
    seq_len: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """Create causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """Create sliding window attention mask."""
    mask = torch.ones(seq_len, seq_len, dtype=dtype, device=device)
    
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 0
    
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    no_decay_keywords: List[str] = ['bias', 'layernorm', 'norm'],
) -> List[Dict]:
    """
    Create parameter groups with different weight decay.
    
    Bias and normalization parameters typically shouldn't have weight decay.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(kw in name.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'lr': lr, 'weight_decay': 0.0},
    ]
