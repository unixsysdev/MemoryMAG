"""
Hash-Hop Data Generation for MemoryMAG Phase 1 Training.

Generates synthetic exact-match retrieval tasks:
- Random key-value pairs (user ID → secret hash)
- Buried in noise/haystack text
- Query requires retrieving the exact hash

This validates that the query projector can address specific memory locations.
The random hashes are incompressible - model must actually use memory.

Usage:
    python generate_hash_hop.py --context_len 64000 --num_samples 10000 --output data/hash_hop_64k.jsonl
"""

import argparse
import json
import random
import string
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm


# Noise text templates for haystack
NOISE_TEMPLATES = [
    "The weather today is {adj} with temperatures around {temp} degrees.",
    "In other news, the stock market {action} by {percent}% yesterday.",
    "Scientists have discovered a new species of {animal} in the {location}.",
    "The latest report shows that {metric} has {change} significantly.",
    "According to experts, the {field} industry will {prediction} next year.",
    "Local authorities announced plans to {action} the {place} area.",
    "A recent study found that {percent}% of people prefer {thing}.",
    "The {event} scheduled for {time} has been {status}.",
    "Market analysts predict {commodity} prices will {direction}.",
    "The government released new guidelines regarding {topic}.",
    "Historical records indicate that {event} occurred in {year}.",
    "Researchers at {institution} published findings about {subject}.",
    "The quarterly earnings report shows {metric} at {value}.",
    "Environmental data suggests {observation} in the {region}.",
    "Infrastructure projects in {city} are {progress}.",
]

ADJECTIVES = ["sunny", "cloudy", "rainy", "mild", "cold", "warm", "windy", "humid"]
ACTIONS = ["increased", "decreased", "stabilized", "fluctuated", "surged", "dropped"]
ANIMALS = ["beetle", "frog", "bird", "fish", "butterfly", "moth", "spider", "lizard"]
LOCATIONS = ["Amazon", "Arctic", "Sahara", "Pacific", "Alps", "Himalayas", "outback"]
METRICS = ["productivity", "efficiency", "output", "growth", "revenue", "engagement"]
FIELDS = ["technology", "healthcare", "finance", "energy", "retail", "manufacturing"]
PREDICTIONS = ["expand", "contract", "transform", "consolidate", "innovate"]
PLACES = ["downtown", "suburban", "industrial", "commercial", "residential"]
THINGS = ["remote work", "electric vehicles", "renewable energy", "digital payments"]
EVENTS = ["conference", "meeting", "launch", "ceremony", "exhibition", "workshop"]
TIMES = ["next week", "this month", "tomorrow", "next quarter", "this Friday"]
STATUSES = ["confirmed", "postponed", "cancelled", "rescheduled", "relocated"]
COMMODITIES = ["oil", "gold", "copper", "wheat", "natural gas", "silver"]
DIRECTIONS = ["rise", "fall", "stabilize", "fluctuate", "plateau"]
TOPICS = ["data privacy", "emissions", "workplace safety", "public health"]
INSTITUTIONS = ["MIT", "Stanford", "Oxford", "Cambridge", "Harvard", "Berkeley"]
SUBJECTS = ["climate patterns", "genetic markers", "neural networks", "quantum states"]
REGIONS = ["coastal areas", "mountain regions", "urban centers", "rural districts"]
CITIES = ["New York", "London", "Tokyo", "Berlin", "Sydney", "Singapore"]
PROGRESS = ["progressing well", "facing delays", "ahead of schedule", "under review"]


def generate_random_hash(length: int = 10) -> str:
    """Generate a random alphanumeric hash."""
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def generate_user_id() -> str:
    """Generate a random user ID."""
    prefix = random.choice(["User", "Agent", "Account", "Client", "Member"])
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{suffix}"


def generate_noise_sentence() -> str:
    """Generate a random noise sentence from templates."""
    template = random.choice(NOISE_TEMPLATES)
    
    # Fill in template variables
    result = template.format(
        adj=random.choice(ADJECTIVES),
        temp=random.randint(10, 95),
        action=random.choice(ACTIONS),
        percent=round(random.uniform(0.1, 15.0), 1),
        animal=random.choice(ANIMALS),
        location=random.choice(LOCATIONS),
        metric=random.choice(METRICS),
        change=random.choice(["increased", "decreased", "changed"]),
        field=random.choice(FIELDS),
        prediction=random.choice(PREDICTIONS),
        place=random.choice(PLACES),
        thing=random.choice(THINGS),
        event=random.choice(EVENTS),
        time=random.choice(TIMES),
        status=random.choice(STATUSES),
        commodity=random.choice(COMMODITIES),
        direction=random.choice(DIRECTIONS),
        topic=random.choice(TOPICS),
        year=random.randint(1900, 2024),
        institution=random.choice(INSTITUTIONS),
        subject=random.choice(SUBJECTS),
        value=f"${random.randint(1, 999)}M",
        observation=random.choice(["warming trends", "declining biodiversity", "rising sea levels"]),
        region=random.choice(REGIONS),
        city=random.choice(CITIES),
        progress=random.choice(PROGRESS),
    )
    
    return result


def generate_noise_text(target_tokens: int, chars_per_token: float = 4.0) -> str:
    """Generate noise text of approximately target token count."""
    target_chars = int(target_tokens * chars_per_token)
    sentences = []
    current_chars = 0
    
    while current_chars < target_chars:
        sentence = generate_noise_sentence()
        sentences.append(sentence)
        current_chars += len(sentence) + 1  # +1 for space
    
    return " ".join(sentences)


def generate_needle(user_id: str, secret: str, style: str = "standard") -> str:
    """Generate needle text containing the key-value pair."""
    templates = {
        "standard": f"CRITICAL DATA: The access token for {user_id} is {secret}.",
        "formal": f"SECURITY NOTICE: User {user_id} has been assigned secret key: {secret}.",
        "system": f"[SYSTEM] Credential update - ID: {user_id}, Key: {secret}",
        "log": f"AUTH_LOG: Generated token {secret} for user {user_id}",
        "config": f"user_credentials: {{ id: \"{user_id}\", token: \"{secret}\" }}",
    }
    return templates.get(style, templates["standard"])


def generate_query(user_id: str) -> str:
    """Generate query asking for the secret."""
    templates = [
        f"Question: What is the access token for {user_id}? Answer:",
        f"Query: Retrieve the secret key for {user_id}. Response:",
        f"What is the credential assigned to {user_id}?",
        f"Find the token for user {user_id}:",
    ]
    return random.choice(templates)


def generate_hash_hop_sample(
    context_tokens: int,
    n_needles: int = 1,
    needle_positions: str = "random",  # "random", "early", "middle", "late"
) -> Dict:
    """
    Generate a single hash-hop sample.
    
    Args:
        context_tokens: Total context length in tokens
        n_needles: Number of key-value pairs to embed
        needle_positions: Where to place needles
        
    Returns:
        Dict with 'context', 'query', 'answer', 'metadata'
    """
    # Generate key-value pairs
    pairs = []
    for _ in range(n_needles):
        user_id = generate_user_id()
        secret = generate_random_hash()
        pairs.append((user_id, secret))
    
    # Calculate noise distribution
    # Reserve ~50 tokens for needle, ~30 for query
    needle_tokens = 50 * n_needles
    query_tokens = 30
    noise_tokens = context_tokens - needle_tokens - query_tokens
    
    # Determine needle positions
    if needle_positions == "early":
        positions = [random.uniform(0.05, 0.2) for _ in range(n_needles)]
    elif needle_positions == "middle":
        positions = [random.uniform(0.4, 0.6) for _ in range(n_needles)]
    elif needle_positions == "late":
        positions = [random.uniform(0.8, 0.95) for _ in range(n_needles)]
    else:  # random
        positions = [random.uniform(0.1, 0.9) for _ in range(n_needles)]
    positions.sort()
    
    # Build context with needles at specified positions
    context_parts = []
    prev_pos = 0.0
    
    for (user_id, secret), pos in zip(pairs, positions):
        # Add noise before needle
        noise_before = int(noise_tokens * (pos - prev_pos))
        if noise_before > 0:
            context_parts.append(generate_noise_text(noise_before))
        
        # Add needle
        style = random.choice(["standard", "formal", "system", "log", "config"])
        needle = generate_needle(user_id, secret, style)
        context_parts.append(needle)
        
        prev_pos = pos
    
    # Add remaining noise
    remaining_noise = int(noise_tokens * (1.0 - prev_pos))
    if remaining_noise > 0:
        context_parts.append(generate_noise_text(remaining_noise))
    
    context = " ".join(context_parts)
    
    # Select query target (random needle if multiple)
    target_idx = random.randint(0, n_needles - 1)
    target_user, target_secret = pairs[target_idx]
    
    query = generate_query(target_user)
    
    return {
        "context": context,
        "query": query,
        "answer": target_secret,
        "metadata": {
            "n_needles": n_needles,
            "target_position": positions[target_idx],
            "context_tokens_approx": context_tokens,
            "all_pairs": [(u, s) for u, s in pairs],
        }
    }


def generate_multi_hop_sample(
    context_tokens: int,
    n_hops: int = 2,
) -> Dict:
    """
    Generate a multi-hop retrieval sample.
    
    Example (2-hop):
    - Needle 1: "User_abc's manager is User_xyz"
    - Needle 2: "User_xyz's access code is qw3rt7"
    - Query: "What is User_abc's manager's access code?"
    - Answer: "qw3rt7"
    """
    # Generate chain of relationships
    users = [generate_user_id() for _ in range(n_hops + 1)]
    final_secret = generate_random_hash()
    
    # Create needles for the chain
    needles = []
    for i in range(n_hops - 1):
        relation = random.choice(["manager", "supervisor", "delegate", "backup"])
        needles.append(f"RECORD: {users[i]}'s {relation} is {users[i+1]}.")
    
    # Final needle has the secret
    needles.append(f"CREDENTIAL: {users[-1]}'s access code is {final_secret}.")
    
    # Calculate token distribution
    needle_tokens = 40 * len(needles)
    query_tokens = 50
    noise_tokens = context_tokens - needle_tokens - query_tokens
    
    # Distribute needles across context
    positions = sorted([random.uniform(0.1, 0.9) for _ in needles])
    
    # Build context
    context_parts = []
    prev_pos = 0.0
    
    for needle, pos in zip(needles, positions):
        noise_before = int(noise_tokens * (pos - prev_pos) / (1.0 - prev_pos + 0.01))
        if noise_before > 0:
            context_parts.append(generate_noise_text(noise_before))
        context_parts.append(needle)
        prev_pos = pos
    
    # Remaining noise
    remaining = noise_tokens - sum(len(p) // 4 for p in context_parts if "RECORD" not in p and "CREDENTIAL" not in p)
    if remaining > 0:
        context_parts.append(generate_noise_text(remaining))
    
    context = " ".join(context_parts)
    
    # Generate query
    chain_desc = "'s manager" * (n_hops - 1) + "'s access code"
    query = f"Question: What is {users[0]}{chain_desc}? Answer:"
    
    return {
        "context": context,
        "query": query,
        "answer": final_secret,
        "metadata": {
            "n_hops": n_hops,
            "chain": users,
            "needle_positions": positions,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Hash-Hop training data")
    parser.add_argument("--context_len", type=int, default=64000, help="Context length in tokens")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/hash_hop.jsonl", help="Output file")
    parser.add_argument("--n_needles", type=int, default=1, help="Needles per sample")
    parser.add_argument("--multi_hop", action="store_true", help="Generate multi-hop samples")
    parser.add_argument("--n_hops", type=int, default=2, help="Number of hops for multi-hop")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--position_dist", type=str, default="random", 
                       choices=["random", "early", "middle", "late"],
                       help="Needle position distribution")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_samples} samples with context length ~{args.context_len} tokens")
    
    with open(output_path, 'w') as f:
        for i in tqdm(range(args.num_samples)):
            if args.multi_hop:
                sample = generate_multi_hop_sample(
                    context_tokens=args.context_len,
                    n_hops=args.n_hops,
                )
            else:
                sample = generate_hash_hop_sample(
                    context_tokens=args.context_len,
                    n_needles=args.n_needles,
                    needle_positions=args.position_dist,
                )
            
            # Format for training
            training_sample = {
                "input": sample["context"] + " " + sample["query"],
                "target": sample["answer"],
                "metadata": sample["metadata"],
            }
            
            f.write(json.dumps(training_sample) + "\n")
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
