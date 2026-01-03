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


class TrainingConfig:
    """Shim for loading checkpoints saved with TrainingConfig pickles."""
    pass


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
        
        # Forward pass - gate values are stored in model's buffer
        self.model.reset_all_memory()
        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get gate values from the model's buffer
        gate_values = {}
        if hasattr(self.model, '_gate_values_buffer'):
            for layer_idx, gates in self.model._gate_values_buffer.items():
                gate_values[layer_idx] = gates.detach().cpu()
        
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
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--patch_layers", type=str, default="all",
                        help="Which layers to patch: 'all', 'every_N', 'last_N', or comma-separated indices")
    parser.add_argument("--memory_layers", type=int, default=2)
    parser.add_argument("--d_memory", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--n_persistent_tokens", type=int, default=16)
    parser.add_argument("--attention_window", type=int, default=None,
                        help="Limit attention to this many previous tokens (forces memory use)")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load base model in 8-bit (bitsandbytes)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load base model in 4-bit (bitsandbytes)")
    
    args = parser.parse_args()
    
    # Load model
    from transformers import AutoTokenizer, AutoConfig
    from src.patch_model import patch_qwen3_with_mag, Qwen3MAGConfig
    
    logger.info(f"Loading model from {args.checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Choose only one of --load_in_8bit or --load_in_4bit")

    # Parse layers_to_patch
    layers_to_patch = None
    if args.patch_layers != "all":
        if args.patch_layers.startswith("every_"):
            n = int(args.patch_layers.split("_")[1])
            model_config = AutoConfig.from_pretrained(args.model_name)
            layers_to_patch = list(range(0, model_config.num_hidden_layers, n))
        elif args.patch_layers.startswith("last_"):
            n = int(args.patch_layers.split("_")[1])
            model_config = AutoConfig.from_pretrained(args.model_name)
            layers_to_patch = list(range(model_config.num_hidden_layers - n, model_config.num_hidden_layers))
        else:
            layers_to_patch = [int(x.strip()) for x in args.patch_layers.split(",")]

    config = Qwen3MAGConfig(
        memory_layers=args.memory_layers,
        d_memory=args.d_memory,
        chunk_size=args.chunk_size,
        n_persistent_tokens=args.n_persistent_tokens,
        attention_window=args.attention_window,
        layers_to_patch=layers_to_patch,
    )
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = patch_qwen3_with_mag(
        model_name_or_path=args.model_name,
        config=config,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Load checkpoint if exists
    checkpoint_path = Path(args.checkpoint) / "checkpoint.pt"
    if checkpoint_path.exists():
        logger.info("Loading checkpoint weights...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
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
