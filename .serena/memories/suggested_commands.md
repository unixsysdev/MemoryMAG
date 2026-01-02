# Suggested Commands

## Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers einops tqdm accelerate

# For Qwen3 models
pip install transformers>=4.40.0
```

## Data Generation

```bash
# Generate synthetic training data
python data/generate_hash_hop.py --context_len 64000 --num_samples 10000
python data/generate_dependency.py --hops 3 --num_samples 10000
python data/generate_code_retrieval.py --repo_path /path/to/repo
```

## Training

```bash
# Train MAG components (freezes base model)
python training/train_mag.py \
    --model_name Qwen/Qwen3-1.7B \
    --data_path data/hash_hop_64k.jsonl \
    --output_dir checkpoints/ \
    --learning_rate 1e-4 \
    --num_epochs 3
```

## Evaluation

```bash
# Run needle-in-haystack benchmark
python evaluation/needle_test.py --checkpoint checkpoints/best --context_len 128000

# Run code completion evaluation
python evaluation/code_completion.py --checkpoint checkpoints/best
```

## Debugging

```bash
# Monitor gate activations
python training/diagnostics.py --checkpoint checkpoints/latest

# Test forward pass only
python -c "from src.patch_model import patch_qwen3_with_mag; m = patch_qwen3_with_mag('Qwen/Qwen3-1.7B'); print('OK')"
```
