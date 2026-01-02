"""
Main Training Script for MemoryMAG.

Implements BPTT-through-TTT (Backpropagation Through Test-Time Training):
- Forward pass through MAG model updates memory weights
- Loss computed on predictions
- Gradients flow back through memory updates
- Teaches query projectors and gates when to use memory

Usage:
    python train_mag.py \
        --model_name Qwen/Qwen3-1.7B \
        --data_path data/hash_hop_64k.jsonl \
        --output_dir checkpoints/ \
        --learning_rate 1e-4 \
        --num_epochs 3
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patch_model import patch_qwen3_with_mag, Qwen3MAGConfig
from src.utils import GateAnalyzer, MemoryDiagnostics, get_parameter_groups

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-1.7B"
    
    # Data
    data_path: str = "data/hash_hop_64k.jsonl"
    eval_data_path: Optional[str] = None
    max_seq_length: int = 4096
    max_samples: Optional[int] = None
    
    # MAG configuration
    memory_layers: int = 2
    d_memory: Optional[int] = None
    n_persistent_tokens: int = 16
    chunk_size: int = 64
    
    # Training
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Output
    output_dir: str = "checkpoints"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # Resume
    resume_from: Optional[str] = None
    
    # Hardware
    device: str = "auto"
    dtype: str = "bfloat16"
    
    # Curriculum
    curriculum_phase: int = 1  # 1=hash_hop, 2=dependency, 3=code


class MemoryMAGDataset(Dataset):
    """Dataset for MemoryMAG training."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer,
        max_seq_length: int = 4096,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        
        logger.info(f"Loading data from {data_path}")
        
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line)
                self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get input and target
        input_text = sample["input"]
        target_text = sample["target"]
        
        # Tokenize
        # For causal LM, we concatenate input and target
        full_text = input_text + " " + target_text
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Create labels (shifted by 1 for causal LM)
        labels = input_ids.clone()
        
        # Mask input portion (only compute loss on target)
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_len = len(input_encoding["input_ids"])
        
        # Set labels to -100 for input tokens (ignored in loss)
        labels[:input_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    # Pad sequences in batch
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class Trainer:
    """MemoryMAG Trainer with BPTT-through-TTT."""
    
    def __init__(
        self,
        model,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        config: TrainingConfig,
        tokenizer,
        use_8bit_optimizer: bool = False,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.tokenizer = tokenizer
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = model.get_trainable_parameters()
        param_groups = get_parameter_groups(model, config.learning_rate, config.weight_decay)
        
        # Filter to only trainable parameters
        for group in param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
        
        # Use 8-bit optimizer to save memory (requires bitsandbytes)
        if use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(param_groups)
                logger.info("Using 8-bit AdamW optimizer (saves ~50% optimizer memory)")
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to standard AdamW. Install with: pip install bitsandbytes")
                self.optimizer = AdamW(param_groups)
        else:
            self.optimizer = AdamW(param_groups)
        
        # Setup scheduler
        total_steps = (len(train_dataset) // config.batch_size // config.gradient_accumulation_steps) * config.num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        # Setup data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
        
        # Diagnostics
        n_layers = model.n_layers
        self.gate_analyzer = GateAnalyzer(n_layers)
        self.memory_diagnostics = MemoryDiagnostics()
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Num epochs: {self.config.num_epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps: {len(self.train_loader) // self.config.gradient_accumulation_steps * self.config.num_epochs}")
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")
        
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)
            
            if self.eval_dataset:
                eval_loss = self.evaluate()
                logger.info(f"Epoch {epoch} eval loss: {eval_loss:.4f}")
                
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best")
            
            self.save_checkpoint(f"epoch_{epoch}")
        
        logger.info("Training complete!")
        self.gate_analyzer.print_summary()
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Reset memory at start of sequence
            self.model.reset_all_memory()
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                    })
                
                # Evaluation
                if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    logger.info(f"Step {self.global_step} eval loss: {eval_loss:.4f}")
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on eval set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.model.reset_all_memory()
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save only trainable parameters
        trainable_state = {}
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_state[param_name] = param.cpu().clone()
        
        torch.save({
            'trainable_state': trainable_state,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load trainable parameters
        model_state = self.model.state_dict()
        for name, param in checkpoint['trainable_state'].items():
            if name in model_state:
                model_state[name].copy_(param)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MemoryMAG")
    
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=None)
    
    # MAG config
    parser.add_argument("--memory_layers", type=int, default=2)
    parser.add_argument("--n_persistent_tokens", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=64)
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    # Resume from previous checkpoint (for curriculum training)
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint dir to resume from (e.g., checkpoints/phase1/best)")
    
    # Hardware
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to reduce memory usage (slower but uses less VRAM)")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                       choices=["eager", "sdpa", "flash_attention_2"],
                       help="Attention implementation: eager (high memory), sdpa (memory efficient), flash_attention_2 (fastest, needs flash-attn)")
    parser.add_argument("--patch_layers", type=str, default="all",
                       help="Which layers to patch: 'all', 'every_N' (e.g. 'every_4'), 'last_N' (e.g. 'last_8'), or comma-separated indices")
    parser.add_argument("--optim_8bit", action="store_true",
                       help="Use 8-bit AdamW optimizer (requires bitsandbytes, saves ~50%% optimizer memory)")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(**{k: v for k, v in vars(args).items() if hasattr(TrainingConfig, k)})
    
    # Setup dtype
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Parse layers_to_patch
    layers_to_patch = None
    if args.patch_layers != "all":
        if args.patch_layers.startswith("every_"):
            # Every Nth layer
            n = int(args.patch_layers.split("_")[1])
            # Will be computed after model load to know total layers
            layers_to_patch = ("every", n)
        elif args.patch_layers.startswith("last_"):
            # Last N layers
            n = int(args.patch_layers.split("_")[1])
            layers_to_patch = ("last", n)
        else:
            # Comma-separated indices
            layers_to_patch = [int(x.strip()) for x in args.patch_layers.split(",")]
    
    # Create MAG config (layers_to_patch will be updated after we know n_layers)
    mag_config = Qwen3MAGConfig(
        memory_layers=args.memory_layers,
        n_persistent_tokens=args.n_persistent_tokens,
        chunk_size=args.chunk_size,
        layers_to_patch=layers_to_patch if isinstance(layers_to_patch, list) else None,
    )
    
    # Load and patch model
    logger.info(f"Loading model: {args.model_name}")
    
    # If using dynamic layer selection, we need to know n_layers first
    if isinstance(layers_to_patch, tuple):
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(args.model_name)
        n_layers = model_config.num_hidden_layers
        mode, n = layers_to_patch
        if mode == "every":
            mag_config.layers_to_patch = list(range(0, n_layers, n))
        elif mode == "last":
            mag_config.layers_to_patch = list(range(n_layers - n, n_layers))
        logger.info(f"Patching layers: {mag_config.layers_to_patch}")
    
    model = patch_qwen3_with_mag(
        model_name_or_path=args.model_name,
        config=mag_config,
        device=args.device,
        dtype=dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        attn_implementation=args.attn_implementation,
    )
    
    # Load datasets
    train_dataset = MemoryMAGDataset(
        args.data_path,
        tokenizer,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
    )
    
    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = MemoryMAGDataset(
            args.eval_data_path,
            tokenizer,
            max_seq_length=args.max_seq_length,
            max_samples=args.max_samples // 10 if args.max_samples else None,
        )
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        tokenizer=tokenizer,
        use_8bit_optimizer=args.optim_8bit,
    )
    
    # Resume from checkpoint if specified (for curriculum continuation)
    if args.resume_from:
        checkpoint_file = Path(args.resume_from) / "checkpoint.pt"
        if checkpoint_file.exists():
            logger.info(f"Resuming from {args.resume_from}")
            trainer.load_checkpoint(str(checkpoint_file))
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_file}, starting fresh")
    
    trainer.train()


if __name__ == "__main__":
    main()
