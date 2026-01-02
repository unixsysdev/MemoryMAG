"""
Code Completion Evaluation for MemoryMAG.

Evaluates real-world utility on code completion tasks.
Tests whether the model can retrieve distant definitions
to complete code correctly.

Usage:
    python code_completion.py --checkpoint checkpoints/best
"""

import argparse
import json
import logging
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Synthetic code completion test cases
TEST_CASES = [
    {
        "description": "Retrieve constant value",
        "setup": '''# config.py
DATABASE_TIMEOUT = 30
MAX_RETRIES = 5
BATCH_SIZE = 64
API_ENDPOINT = "https://api.example.com"
''',
        "prompt": '''# worker.py
def process_batch(items):
    """Process items in batches."""
    for i in range(0, len(items), ''',
        "expected_contains": ["BATCH_SIZE", "64"],
        "context_padding": 50000,
    },
    {
        "description": "Retrieve function signature",
        "setup": '''# utils.py
def validate_email(email: str, strict: bool = False) -> bool:
    """Validate email format.
    
    Args:
        email: Email address to validate
        strict: If True, use strict validation rules
        
    Returns:
        True if valid, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if strict:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$'
    return bool(re.match(pattern, email))
''',
        "prompt": '''# registration.py
def register_user(email, password):
    # First validate the email
    if not ''',
        "expected_contains": ["validate_email"],
        "context_padding": 40000,
    },
    {
        "description": "Retrieve class for instantiation",
        "setup": '''# models.py
class DatabaseConnection:
    """Manages database connections."""
    
    def __init__(self, host: str, port: int = 5432, timeout: int = 30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._connection = None
    
    def connect(self):
        """Establish database connection."""
        # Implementation here
        pass
    
    def execute(self, query: str):
        """Execute a query."""
        pass
''',
        "prompt": '''# service.py
def get_users():
    # Create database connection
    db = ''',
        "expected_contains": ["DatabaseConnection"],
        "context_padding": 60000,
    },
    {
        "description": "Retrieve enum value",
        "setup": '''# types.py
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
''',
        "prompt": '''# handlers.py
def mark_complete(task):
    """Mark task as completed."""
    task.status = Status.''',
        "expected_contains": ["COMPLETED"],
        "context_padding": 30000,
    },
]


def generate_padding(target_tokens: int) -> str:
    """Generate code-like padding text."""
    templates = [
        '''def helper_{n}(x):
    """Helper function {n}."""
    return x * {m}

''',
        '''class Handler{n}:
    """Handler for type {n}."""
    def process(self, data):
        return data

''',
        '''CONFIG_{n} = {{
    "enabled": True,
    "value": {m}
}}

''',
        '''# Processing step {n}
result_{n} = process(input_{m})

''',
    ]
    
    result = []
    current_chars = 0
    target_chars = target_tokens * 4
    n = 0
    
    while current_chars < target_chars:
        template = random.choice(templates)
        code = template.format(n=n, m=random.randint(1, 100))
        result.append(code)
        current_chars += len(code)
        n += 1
    
    return "".join(result)


class CodeCompletionBenchmark:
    """
    Code completion benchmark for evaluating semantic retrieval.
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def run_test(
        self,
        setup: str,
        prompt: str,
        expected_contains: List[str],
        context_padding: int,
        max_new_tokens: int = 30,
    ) -> Dict:
        """Run a single code completion test."""
        # Generate padding
        padding = generate_padding(context_padding)
        
        # Build full context: setup + padding + prompt
        full_context = f"{setup}\n\n{padding}\n\n{prompt}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_context,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Reset memory
        self.model.reset_all_memory()
        
        # Generate completion
        output_ids = self.model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode
        generated_ids = output_ids[0, input_ids.shape[1]:]
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Check if expected content is present
        success = any(exp.lower() in completion.lower() for exp in expected_contains)
        
        return {
            "completion": completion,
            "expected": expected_contains,
            "success": success,
            "context_tokens": input_ids.shape[1],
        }
    
    def run_benchmark(self, test_cases: List[Dict] = None) -> Dict:
        """Run full benchmark."""
        if test_cases is None:
            test_cases = TEST_CASES
        
        results = {
            "tests": [],
            "overall": {"success": 0, "total": 0},
        }
        
        for test in tqdm(test_cases, desc="Running code completion tests"):
            try:
                result = self.run_test(
                    setup=test["setup"],
                    prompt=test["prompt"],
                    expected_contains=test["expected_contains"],
                    context_padding=test["context_padding"],
                )
                
                result["description"] = test["description"]
                results["tests"].append(result)
                
                results["overall"]["total"] += 1
                if result["success"]:
                    results["overall"]["success"] += 1
                    
            except Exception as e:
                logger.warning(f"Test failed: {test['description']} - {e}")
                results["tests"].append({
                    "description": test["description"],
                    "error": str(e),
                    "success": False,
                })
                results["overall"]["total"] += 1
        
        results["overall"]["accuracy"] = (
            results["overall"]["success"] / results["overall"]["total"]
            if results["overall"]["total"] > 0 else 0
        )
        
        return results
    
    def print_results(self, results: Dict):
        """Print benchmark results."""
        print("\n" + "=" * 70)
        print("CODE COMPLETION BENCHMARK RESULTS")
        print("=" * 70)
        
        print(f"\n📊 OVERALL: {results['overall']['accuracy']:.1%} "
              f"({results['overall']['success']}/{results['overall']['total']})")
        
        print("\n📝 INDIVIDUAL TESTS:")
        print("-" * 50)
        
        for test in results["tests"]:
            status = "✓" if test.get("success") else "✗"
            desc = test.get("description", "Unknown")
            
            print(f"\n  {status} {desc}")
            
            if "completion" in test:
                completion_preview = test["completion"][:80].replace("\n", " ")
                print(f"    Completion: {completion_preview}...")
                print(f"    Expected: {test['expected']}")
                
                if "context_tokens" in test:
                    print(f"    Context: {test['context_tokens']:,} tokens")
            
            if "error" in test:
                print(f"    Error: {test['error']}")
        
        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run code completion benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    from transformers import AutoTokenizer
    from src.patch_model import patch_qwen3_with_mag, Qwen3MAGConfig
    
    logger.info(f"Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = Qwen3MAGConfig()
    model = patch_qwen3_with_mag(
        model_name_or_path=args.model_name,
        config=config,
        device=args.device,
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) / "checkpoint.pt"
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model_state = model.state_dict()
        for name, param in checkpoint['trainable_state'].items():
            if name in model_state:
                model_state[name].copy_(param)
    
    # Run benchmark
    benchmark = CodeCompletionBenchmark(model, tokenizer, device=args.device)
    results = benchmark.run_benchmark()
    
    # Print and save
    benchmark.print_results(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
