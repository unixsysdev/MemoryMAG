"""
Needle-in-Haystack Benchmark for MemoryMAG.

Tests the model's ability to retrieve specific information buried in long context.
This is the primary evaluation for memory retrieval capability.

Metrics:
- Exact match accuracy at various context lengths
- Retrieval accuracy by needle position (early, middle, late)
- Performance degradation curve as context increases

Usage:
    python needle_test.py --checkpoint checkpoints/best --context_len 128000
"""

import argparse
import json
import logging
import sys
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from tqdm import tqdm

# Add src/data to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))


class TrainingConfig:
    """Placeholder to load checkpoints saved with training.TrainingConfig in __main__."""
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NeedleTestResult:
    """Result from a single needle test."""
    context_length: int
    needle_position: float  # 0.0 = start, 1.0 = end
    needle_text: str
    expected_answer: str
    model_answer: str
    correct: bool
    tokens_generated: int


def generate_haystack(target_tokens: int, chars_per_token: float = 4.0) -> str:
    """Generate random haystack text."""
    target_chars = int(target_tokens * chars_per_token)
    
    sentences = [
        "The weather today is quite pleasant with mild temperatures.",
        "Scientists continue to make progress in various research fields.",
        "Economic indicators suggest stable growth in the current quarter.",
        "Technology advances are reshaping how we interact with the world.",
        "Environmental concerns drive policy changes across nations.",
        "Healthcare innovations promise improved patient outcomes.",
        "Educational institutions adapt to changing societal needs.",
        "Infrastructure development remains a priority for governments.",
        "Cultural exchanges foster understanding between peoples.",
        "Scientific discoveries open new avenues for exploration.",
    ]
    
    result = []
    current_chars = 0
    
    while current_chars < target_chars:
        sentence = random.choice(sentences)
        result.append(sentence)
        current_chars += len(sentence) + 1
    
    return " ".join(result)


def generate_needle(secret_key: str) -> Tuple[str, str, str]:
    """
    Generate needle text and corresponding query.
    
    Returns:
        needle_text: Text to embed in haystack
        query: Question to ask
        answer: Expected answer
    """
    user_id = f"User_{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"
    
    needle_templates = [
        f"IMPORTANT: The secret access code for {user_id} is {secret_key}.",
        f"NOTICE: User {user_id} has been assigned token {secret_key}.",
        f"[SYSTEM] Credential for {user_id}: {secret_key}",
    ]
    
    query_templates = [
        f"What is the secret access code for {user_id}?",
        f"What token was assigned to {user_id}?",
        f"What is the credential for {user_id}?",
    ]
    
    needle = random.choice(needle_templates)
    query = random.choice(query_templates)
    
    return needle, query, secret_key


def create_needle_test(
    context_tokens: int,
    needle_position: float,
) -> Tuple[str, str, str]:
    """
    Create a needle-in-haystack test case.
    
    Args:
        context_tokens: Total context length
        needle_position: Where to place needle (0.0 to 1.0)
        
    Returns:
        context: Full context with needle embedded
        query: Question to ask
        expected_answer: Correct answer
    """
    # Generate random answer
    secret_key = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    
    # Generate needle
    needle, query, answer = generate_needle(secret_key)
    needle_tokens = len(needle) // 4
    
    # Calculate haystack distribution
    tokens_before = int((context_tokens - needle_tokens) * needle_position)
    tokens_after = context_tokens - needle_tokens - tokens_before
    
    # Generate haystacks
    haystack_before = generate_haystack(tokens_before)
    haystack_after = generate_haystack(tokens_after)
    
    # Combine
    context = f"{haystack_before} {needle} {haystack_after}"
    
    return context, query, answer


class NeedleBenchmark:
    """
    Needle-in-Haystack benchmark runner.
    
    Tests model across:
    - Multiple context lengths
    - Various needle positions
    - Multiple repetitions for statistical significance
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def run_single_test(
        self,
        context: str,
        query: str,
        expected_answer: str,
        max_new_tokens: int = 50,
    ) -> NeedleTestResult:
        """Run a single needle test."""
        # Prepare input
        full_input = f"{context}\n\nQuestion: {query}\nAnswer:"
        
        encoding = self.tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        context_length = input_ids.shape[1]
        
        # Reset memory before generation
        self.model.reset_all_memory()
        
        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode response
        generated_ids = output_ids[0, input_ids.shape[1]:]
        model_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Check correctness (exact match or contains)
        correct = (
            expected_answer.lower() in model_answer.lower() or
            model_answer.lower().startswith(expected_answer.lower())
        )
        
        return NeedleTestResult(
            context_length=context_length,
            needle_position=0.5,  # Will be set by caller
            needle_text="",
            expected_answer=expected_answer,
            model_answer=model_answer,
            correct=correct,
            tokens_generated=len(generated_ids),
        )
    
    def run_benchmark(
        self,
        context_lengths: List[int] = [2000, 4000, 8000, 16000, 32000],
        needle_positions: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        n_trials: int = 5,
    ) -> Dict:
        """
        Run full benchmark suite.
        
        Args:
            context_lengths: List of context lengths to test
            needle_positions: List of positions (0.0 to 1.0)
            n_trials: Number of trials per condition
            
        Returns:
            Benchmark results dictionary
        """
        results = {
            "by_length": {},
            "by_position": {},
            "overall": {},
            "raw_results": [],
        }
        
        total_tests = len(context_lengths) * len(needle_positions) * n_trials
        pbar = tqdm(total=total_tests, desc="Running needle tests")
        
        for ctx_len in context_lengths:
            results["by_length"][ctx_len] = {"correct": 0, "total": 0}
            
            for position in needle_positions:
                pos_key = f"{position:.2f}"
                if pos_key not in results["by_position"]:
                    results["by_position"][pos_key] = {"correct": 0, "total": 0}
                
                for trial in range(n_trials):
                    # Generate test case
                    context, query, expected = create_needle_test(ctx_len, position)
                    
                    # Run test
                    try:
                        result = self.run_single_test(context, query, expected)
                        result.needle_position = position
                        
                        # Update statistics
                        results["by_length"][ctx_len]["total"] += 1
                        results["by_position"][pos_key]["total"] += 1
                        
                        if result.correct:
                            results["by_length"][ctx_len]["correct"] += 1
                            results["by_position"][pos_key]["correct"] += 1
                        
                        results["raw_results"].append({
                            "context_length": ctx_len,
                            "position": position,
                            "expected": expected,
                            "model_answer": result.model_answer,
                            "correct": result.correct,
                        })
                    except Exception as e:
                        logger.warning(f"Test failed: {e}")
                    
                    pbar.update(1)
        
        pbar.close()
        
        # Calculate overall accuracy
        total_correct = sum(r["correct"] for r in results["by_length"].values())
        total_tests = sum(r["total"] for r in results["by_length"].values())
        
        results["overall"] = {
            "accuracy": total_correct / total_tests if total_tests > 0 else 0,
            "total_tests": total_tests,
            "total_correct": total_correct,
        }
        
        # Calculate per-condition accuracies
        for ctx_len in results["by_length"]:
            stats = results["by_length"][ctx_len]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        for pos in results["by_position"]:
            stats = results["by_position"][pos]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        
        return results
    
    def print_results(self, results: Dict):
        """Print benchmark results."""
        print("\n" + "=" * 70)
        print("NEEDLE-IN-HAYSTACK BENCHMARK RESULTS")
        print("=" * 70)
        
        print(f"\n📊 OVERALL: {results['overall']['accuracy']:.1%} "
              f"({results['overall']['total_correct']}/{results['overall']['total_tests']})")
        
        print("\n📏 BY CONTEXT LENGTH:")
        print("-" * 40)
        for ctx_len in sorted(results["by_length"].keys()):
            stats = results["by_length"][ctx_len]
            bar = "█" * int(stats["accuracy"] * 20) + "░" * (20 - int(stats["accuracy"] * 20))
            print(f"  {ctx_len:6,} tokens: {bar} {stats['accuracy']:.1%}")
        
        print("\n📍 BY NEEDLE POSITION:")
        print("-" * 40)
        for pos in sorted(results["by_position"].keys()):
            stats = results["by_position"][pos]
            pos_name = {
                "0.10": "Early (10%)",
                "0.25": "Quarter (25%)",
                "0.50": "Middle (50%)",
                "0.75": "Three-quarter (75%)",
                "0.90": "Late (90%)",
            }.get(pos, f"Position {pos}")
            bar = "█" * int(stats["accuracy"] * 20) + "░" * (20 - int(stats["accuracy"] * 20))
            print(f"  {pos_name:20s}: {bar} {stats['accuracy']:.1%}")
        
        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run needle-in-haystack benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--context_lengths", type=str, default="2000,4000,8000,16000")
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_8bit", action="store_true", help="Load base model in 8-bit (bitsandbytes)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit (bitsandbytes)")
    parser.add_argument("--memory_layers", type=int, default=2)
    parser.add_argument("--d_memory", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--n_persistent_tokens", type=int, default=16)
    parser.add_argument("--attention_window", type=int, default=None,
                        help="Limit attention to this many previous tokens (forces memory use)")
    parser.add_argument("--patch_layers", type=str, default="all",
                        help="Which layers to patch: 'all', 'every_N', 'last_N', or comma-separated indices")
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"])
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Parse context lengths
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    
    # Load model
    from transformers import AutoTokenizer, AutoConfig
    from src.patch_model import patch_qwen3_with_mag, Qwen3MAGConfig
    
    logger.info(f"Loading model...")
    
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
    model = patch_qwen3_with_mag(
        model_name_or_path=args.model_name,
        config=config,
        device=args.device,
        attn_implementation=args.attn_implementation,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) / "checkpoint.pt"
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
        model_state = model.state_dict()
        for name, param in checkpoint['trainable_state'].items():
            if name in model_state:
                model_state[name].copy_(param)
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using base model")
    
    # Run benchmark
    benchmark = NeedleBenchmark(model, tokenizer, device=args.device)
    results = benchmark.run_benchmark(
        context_lengths=context_lengths,
        n_trials=args.n_trials,
    )
    
    # Print and save results
    benchmark.print_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
