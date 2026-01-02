"""
Diagnostics and Monitoring for MemoryMAG Training.

Provides tools to:
- Monitor gate activations (are gates opening appropriately?)
- Track surprise distributions (is memory learning?)
- Visualize query-memory alignment
- Detect training issues early

Usage:
    python diagnostics.py --checkpoint checkpoints/latest
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Result from diagnostic analysis."""
    metric: str
    value: float
    status: str  # 'healthy', 'warning', 'critical'
    message: str


class MAGDiagnostics:
    """
    Comprehensive diagnostics for MAG model.
    
    Analyzes:
    - Gate behavior across layers
    - Memory utilization
    - Query-memory alignment
    - Gradient flow
    """
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Collect layer info
        self.n_layers = model.n_layers
        self.mag_layers = [l for l in model.mag_layers if l is not None]
    
    @torch.no_grad()
    def analyze_gates(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """
        Analyze gate activations across layers.
        
        Returns gate statistics per layer.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # We need to hook into the gate values
        gate_values = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if len(output) >= 3:  # output contains gate_values
                    gate_values[layer_idx] = output[2].detach().cpu()
            return hook
        
        # Register hooks
        handles = []
        for idx, layer in enumerate(self.mag_layers):
            if layer is not None:
                handle = layer.register_forward_hook(make_hook(idx))
                handles.append(handle)
        
        # Forward pass
        self.model.reset_all_memory()
        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Analyze gate values
        results = {
            "per_layer": {},
            "overall": {},
            "diagnostics": [],
        }
        
        all_means = []
        all_stds = []
        
        for layer_idx, gates in gate_values.items():
            mean_gate = gates.mean().item()
            std_gate = gates.std().item()
            min_gate = gates.min().item()
            max_gate = gates.max().item()
            
            all_means.append(mean_gate)
            all_stds.append(std_gate)
            
            results["per_layer"][layer_idx] = {
                "mean": mean_gate,
                "std": std_gate,
                "min": min_gate,
                "max": max_gate,
            }
            
            # Check for issues
            if std_gate < 0.01:
                if mean_gate < 0.1:
                    results["diagnostics"].append(DiagnosticResult(
                        metric=f"gate_layer_{layer_idx}",
                        value=mean_gate,
                        status="warning",
                        message=f"Layer {layer_idx} gate collapsed to ~0 (ignoring memory)",
                    ))
                elif mean_gate > 0.9:
                    results["diagnostics"].append(DiagnosticResult(
                        metric=f"gate_layer_{layer_idx}",
                        value=mean_gate,
                        status="warning",
                        message=f"Layer {layer_idx} gate collapsed to ~1 (ignoring attention)",
                    ))
        
        results["overall"] = {
            "mean_across_layers": sum(all_means) / len(all_means) if all_means else 0,
            "std_across_layers": sum(all_stds) / len(all_stds) if all_stds else 0,
        }
        
        return results
    
    @torch.no_grad()
    def analyze_memory(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """
        Analyze memory behavior.
        
        Returns surprise distributions and memory weight statistics.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        results = {
            "per_layer": {},
            "diagnostics": [],
        }
        
        # Analyze each MAG layer's memory
        for idx, layer in enumerate(self.mag_layers):
            if layer is None:
                continue
            
            memory = layer.neural_memory
            
            # Get memory weight norms
            weight_norms = []
            for mem_layer in memory.memory_layers:
                weight_norms.append(mem_layer.weight.norm().item())
            
            results["per_layer"][idx] = {
                "weight_norms": weight_norms,
                "total_weight_norm": sum(weight_norms),
            }
            
            # Check for issues
            if sum(weight_norms) > 1000:
                results["diagnostics"].append(DiagnosticResult(
                    metric=f"memory_weights_layer_{idx}",
                    value=sum(weight_norms),
                    status="warning",
                    message=f"Layer {idx} memory weights are large (may need more decay)",
                ))
            
            if sum(weight_norms) < 0.1:
                results["diagnostics"].append(DiagnosticResult(
                    metric=f"memory_weights_layer_{idx}",
                    value=sum(weight_norms),
                    status="warning",
                    message=f"Layer {idx} memory weights are very small (may be under-training)",
                ))
        
        return results
    
    @torch.no_grad()
    def analyze_retrieval_quality(
        self,
        input_ids: torch.Tensor,
        needle_positions: List[int],
        query_position: int,
    ) -> Dict[str, any]:
        """
        Analyze whether memory successfully retrieves needle information.
        
        Args:
            input_ids: Input sequence containing needle(s)
            needle_positions: Token positions where needles are located
            query_position: Token position of the query
            
        Returns:
            Analysis of retrieval success
        """
        input_ids = input_ids.to(self.device)
        
        # This is a simplified analysis - in practice you'd want to
        # look at attention patterns and memory activations more carefully
        
        results = {
            "needle_positions": needle_positions,
            "query_position": query_position,
            "analysis": "TODO: Implement detailed retrieval analysis",
        }
        
        return results
    
    def run_full_diagnostic(
        self,
        test_input_ids: torch.Tensor,
        test_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """Run complete diagnostic suite."""
        logger.info("Running full MAG diagnostics...")
        
        results = {
            "gates": self.analyze_gates(test_input_ids, test_attention_mask),
            "memory": self.analyze_memory(test_input_ids, test_attention_mask),
        }
        
        # Collect all diagnostics
        all_diagnostics = []
        all_diagnostics.extend(results["gates"]["diagnostics"])
        all_diagnostics.extend(results["memory"]["diagnostics"])
        
        results["summary"] = {
            "total_warnings": sum(1 for d in all_diagnostics if d.status == "warning"),
            "total_critical": sum(1 for d in all_diagnostics if d.status == "critical"),
            "health": "healthy" if not all_diagnostics else "needs_attention",
        }
        
        return results
    
    def print_report(self, results: Dict[str, any]):
        """Print diagnostic report."""
        print("\n" + "=" * 70)
        print("MEMORYMAG DIAGNOSTIC REPORT")
        print("=" * 70)
        
        # Gate analysis
        print("\n📊 GATE ANALYSIS")
        print("-" * 40)
        gate_results = results["gates"]
        
        for layer_idx, stats in gate_results["per_layer"].items():
            status = "✓" if stats["std"] > 0.01 else "⚠️"
            print(f"  Layer {layer_idx:2d}: mean={stats['mean']:.3f} "
                  f"std={stats['std']:.3f} range=[{stats['min']:.3f}, {stats['max']:.3f}] {status}")
        
        print(f"\n  Overall mean: {gate_results['overall']['mean_across_layers']:.3f}")
        
        # Memory analysis
        print("\n🧠 MEMORY ANALYSIS")
        print("-" * 40)
        memory_results = results["memory"]
        
        for layer_idx, stats in memory_results["per_layer"].items():
            print(f"  Layer {layer_idx:2d}: weight_norm={stats['total_weight_norm']:.2f}")
        
        # Diagnostics
        print("\n⚕️  DIAGNOSTICS")
        print("-" * 40)
        
        all_diagnostics = []
        all_diagnostics.extend(gate_results.get("diagnostics", []))
        all_diagnostics.extend(memory_results.get("diagnostics", []))
        
        if not all_diagnostics:
            print("  ✓ No issues detected")
        else:
            for diag in all_diagnostics:
                icon = "⚠️" if diag.status == "warning" else "❌"
                print(f"  {icon} {diag.message}")
        
        # Summary
        print("\n📋 SUMMARY")
        print("-" * 40)
        summary = results["summary"]
        print(f"  Warnings: {summary['total_warnings']}")
        print(f"  Critical: {summary['total_critical']}")
        print(f"  Status: {summary['health'].upper()}")
        
        print("\n" + "=" * 70)


def generate_diagnostic_input(tokenizer, length: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a simple diagnostic input."""
    # Create a simple test sequence
    text = "This is a test sequence for diagnostic purposes. " * (length // 10)
    
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=length,
        return_tensors="pt",
    )
    
    return encoding["input_ids"], encoding["attention_mask"]


def main():
    parser = argparse.ArgumentParser(description="Run MAG diagnostics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seq_length", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load model
    from transformers import AutoTokenizer
    from src.patch_model import patch_qwen3_with_mag, Qwen3MAGConfig
    
    logger.info(f"Loading model from {args.checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = Qwen3MAGConfig()
    model = patch_qwen3_with_mag(
        model_name_or_path=args.model_name,
        config=config,
        device=args.device,
    )
    
    # Load checkpoint if exists
    checkpoint_path = Path(args.checkpoint) / "checkpoint.pt"
    if checkpoint_path.exists():
        logger.info("Loading checkpoint weights...")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model_state = model.state_dict()
        for name, param in checkpoint['trainable_state'].items():
            if name in model_state:
                model_state[name].copy_(param)
    
    # Run diagnostics
    diagnostics = MAGDiagnostics(model, device=args.device)
    
    input_ids, attention_mask = generate_diagnostic_input(tokenizer, args.seq_length)
    
    results = diagnostics.run_full_diagnostic(input_ids, attention_mask)
    diagnostics.print_report(results)


if __name__ == "__main__":
    main()
