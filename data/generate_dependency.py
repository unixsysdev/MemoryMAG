"""
Dependency Tracing Data Generation for MemoryMAG Phase 2 Training.

Generates multi-hop reasoning tasks that require combining multiple stored facts:
- Chains of relationships (A depends on B, B requires C, ...)
- Facts distributed across long context
- Query requires traversing the dependency chain

This teaches the model query refinement across layers:
- Early layers: Find initial entity
- Middle layers: Follow relationships
- Late layers: Retrieve final answer

Usage:
    python generate_dependency.py --hops 3 --num_samples 10000 --output data/dependency_3hop.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


# Entity types and their relationships
ENTITY_TYPES = {
    "module": {
        "names": ["AuthModule", "DataModule", "APIModule", "CoreModule", "UtilModule", 
                  "NetworkModule", "CacheModule", "LogModule", "ConfigModule", "StorageModule"],
        "relations": ["depends on", "imports", "requires", "extends", "uses"],
    },
    "agent": {
        "names": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"],
        "relations": ["supervises", "reports to", "collaborates with", "delegates to"],
    },
    "service": {
        "names": ["UserService", "OrderService", "PaymentService", "NotificationService",
                  "AnalyticsService", "SearchService", "ReportService", "CacheService"],
        "relations": ["calls", "subscribes to", "publishes to", "depends on"],
    },
    "component": {
        "names": ["Button", "Form", "Modal", "Header", "Footer", "Sidebar", "Card", "Table"],
        "relations": ["renders", "contains", "wraps", "extends"],
    },
    "database": {
        "names": ["UserDB", "OrderDB", "ProductDB", "AnalyticsDB", "CacheDB", "LogDB"],
        "relations": ["replicates to", "syncs with", "reads from", "writes to"],
    },
}

# Final fact types (what the chain leads to)
FINAL_FACTS = {
    "module": {
        "property": "version",
        "values": lambda: f"v{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,99)}",
    },
    "agent": {
        "property": "clearance level",
        "values": lambda: random.choice(["Alpha", "Beta", "Gamma", "Delta", "Omega"]),
    },
    "service": {
        "property": "port",
        "values": lambda: str(random.randint(3000, 9999)),
    },
    "component": {
        "property": "style theme",
        "values": lambda: random.choice(["dark", "light", "ocean", "forest", "sunset"]),
    },
    "database": {
        "property": "shard count",
        "values": lambda: str(random.randint(1, 128)),
    },
}

# Noise generators
NOISE_TOPICS = [
    "system performance", "user feedback", "error rates", "latency metrics",
    "deployment status", "test coverage", "code quality", "security audit",
]


def generate_noise_fact() -> str:
    """Generate a noise fact that looks similar but is irrelevant."""
    entity_type = random.choice(list(ENTITY_TYPES.keys()))
    config = ENTITY_TYPES[entity_type]
    
    entity1 = random.choice(config["names"])
    entity2 = random.choice(config["names"])
    while entity2 == entity1:
        entity2 = random.choice(config["names"])
    
    relation = random.choice(config["relations"])
    
    templates = [
        f"Note: {entity1} {relation} {entity2}.",
        f"Record: The {entity_type} {entity1} {relation} {entity2}.",
        f"[INFO] {entity1} -> {relation} -> {entity2}",
        f"Documentation: {entity1} {relation} {entity2} as of last update.",
    ]
    
    return random.choice(templates)


def generate_noise_text(target_tokens: int) -> str:
    """Generate noise text of approximately target token count."""
    target_chars = int(target_tokens * 4)
    parts = []
    current_chars = 0
    
    while current_chars < target_chars:
        if random.random() < 0.7:
            # Generate a noise fact
            fact = generate_noise_fact()
        else:
            # Generate generic noise
            topic = random.choice(NOISE_TOPICS)
            value = random.randint(1, 100)
            fact = f"The {topic} shows {value}% {random.choice(['improvement', 'stability', 'change'])}."
        
        parts.append(fact)
        current_chars += len(fact) + 1
    
    return " ".join(parts)


def generate_dependency_chain(
    n_hops: int,
    entity_type: Optional[str] = None,
) -> Tuple[List[str], List[str], str, str]:
    """
    Generate a chain of dependencies.
    
    Returns:
        entities: List of entity names in the chain
        relations: List of relations connecting them
        final_property: The property asked about
        final_value: The answer
    """
    if entity_type is None:
        entity_type = random.choice(list(ENTITY_TYPES.keys()))
    
    config = ENTITY_TYPES[entity_type]
    
    # Generate unique entities for the chain
    available = config["names"].copy()
    random.shuffle(available)
    
    if len(available) < n_hops + 1:
        # Need more entities, add suffixes
        base_entities = available.copy()
        for i in range(n_hops + 1 - len(available)):
            available.append(f"{random.choice(base_entities)}_{i+1}")
    
    entities = available[:n_hops + 1]
    
    # Generate relations
    relations = [random.choice(config["relations"]) for _ in range(n_hops)]
    
    # Generate final fact
    final_config = FINAL_FACTS[entity_type]
    final_property = final_config["property"]
    final_value = final_config["values"]()
    
    return entities, relations, final_property, final_value


def create_chain_facts(
    entities: List[str],
    relations: List[str],
    final_property: str,
    final_value: str,
) -> List[str]:
    """Create fact strings for the dependency chain."""
    facts = []
    
    # Relationship facts
    for i, (e1, rel, e2) in enumerate(zip(entities[:-1], relations, entities[1:])):
        templates = [
            f"DEPENDENCY: {e1} {rel} {e2}.",
            f"[LINK] {e1} -> {rel} -> {e2}",
            f"Fact: {e1} {rel} {e2}.",
            f"Record: {e1} has relationship '{rel}' with {e2}.",
        ]
        facts.append(random.choice(templates))
    
    # Final fact
    final_entity = entities[-1]
    final_templates = [
        f"PROPERTY: {final_entity}'s {final_property} is {final_value}.",
        f"[DATA] {final_entity}.{final_property} = {final_value}",
        f"Info: The {final_property} of {final_entity} is {final_value}.",
        f"Attribute: {final_entity} has {final_property}: {final_value}.",
    ]
    facts.append(random.choice(final_templates))
    
    return facts


def generate_query(
    entities: List[str],
    relations: List[str],
    final_property: str,
) -> str:
    """Generate query for the dependency chain."""
    start_entity = entities[0]
    
    # Build relationship chain description
    if len(relations) == 1:
        chain_desc = f"what {start_entity} {relations[0]}"
    elif len(relations) == 2:
        chain_desc = f"what {start_entity} {relations[0]}, and then what that {relations[1]}"
    else:
        chain_desc = f"following the chain from {start_entity}"
    
    templates = [
        f"Question: What is the {final_property} of the entity that {chain_desc}? Answer:",
        f"Query: Starting from {start_entity}, follow the dependency chain. What is the final entity's {final_property}?",
        f"Task: Trace dependencies starting from {start_entity}. Report the {final_property} of the final entity.",
        f"Find: {start_entity} -> ... -> ? What is the {final_property}?",
    ]
    
    return random.choice(templates)


def generate_dependency_sample(
    context_tokens: int,
    n_hops: int = 2,
    add_distractors: bool = True,
) -> Dict:
    """
    Generate a dependency tracing sample.
    
    Args:
        context_tokens: Total context length in tokens
        n_hops: Number of hops in the dependency chain
        add_distractors: Add distractor chains that are similar but not queried
        
    Returns:
        Dict with training data
    """
    # Generate main chain
    entity_type = random.choice(list(ENTITY_TYPES.keys()))
    entities, relations, final_prop, final_value = generate_dependency_chain(n_hops, entity_type)
    
    # Create facts for main chain
    main_facts = create_chain_facts(entities, relations, final_prop, final_value)
    
    # Add distractor chains if requested
    distractor_facts = []
    if add_distractors:
        n_distractors = random.randint(2, 5)
        for _ in range(n_distractors):
            d_entities, d_relations, d_prop, d_value = generate_dependency_chain(
                random.randint(1, n_hops), entity_type
            )
            distractor_facts.extend(create_chain_facts(d_entities, d_relations, d_prop, d_value))
    
    # All facts to distribute
    all_facts = main_facts + distractor_facts
    random.shuffle(all_facts)
    
    # Calculate token distribution
    fact_tokens = sum(len(f) // 4 for f in all_facts)
    query_tokens = 50
    noise_tokens = context_tokens - fact_tokens - query_tokens
    
    # Distribute facts across context
    n_facts = len(all_facts)
    positions = sorted([random.uniform(0.1, 0.9) for _ in range(n_facts)])
    
    # Build context
    context_parts = []
    prev_pos = 0.0
    
    for fact, pos in zip(all_facts, positions):
        # Add noise before fact
        noise_portion = (pos - prev_pos) / (1.0 - prev_pos + 0.01)
        noise_for_gap = int(noise_tokens * noise_portion)
        if noise_for_gap > 10:
            context_parts.append(generate_noise_text(noise_for_gap))
        
        context_parts.append(fact)
        prev_pos = pos
    
    # Remaining noise
    remaining = max(100, noise_tokens - sum(len(p) // 4 for p in context_parts))
    context_parts.append(generate_noise_text(remaining))
    
    context = " ".join(context_parts)
    query = generate_query(entities, relations, final_prop)
    
    return {
        "context": context,
        "query": query,
        "answer": final_value,
        "metadata": {
            "n_hops": n_hops,
            "entity_type": entity_type,
            "chain": entities,
            "relations": relations,
            "final_property": final_prop,
            "n_distractors": len(distractor_facts),
        }
    }


def generate_complex_reasoning_sample(
    context_tokens: int,
    n_chains: int = 2,
    hops_per_chain: int = 2,
) -> Dict:
    """
    Generate a sample requiring combination of multiple chains.
    
    Example:
    - Chain 1: A -> B -> C (C has property X)
    - Chain 2: D -> E -> C (same C!)
    - Query: "What property X does the entity that both A and D lead to have?"
    """
    entity_type = random.choice(list(ENTITY_TYPES.keys()))
    config = ENTITY_TYPES[entity_type]
    
    # Generate shared endpoint
    shared_entity = random.choice(config["names"])
    final_prop = FINAL_FACTS[entity_type]["property"]
    final_value = FINAL_FACTS[entity_type]["values"]()
    
    # Generate chains leading to shared entity
    all_facts = []
    chain_starts = []
    
    for chain_idx in range(n_chains):
        # Generate chain leading to shared entity
        available = [n for n in config["names"] if n != shared_entity]
        random.shuffle(available)
        
        chain = available[:hops_per_chain] + [shared_entity]
        relations = [random.choice(config["relations"]) for _ in range(hops_per_chain)]
        
        chain_starts.append(chain[0])
        
        # Create facts (without final property - added once)
        for i, (e1, rel, e2) in enumerate(zip(chain[:-1], relations, chain[1:])):
            fact = f"LINK: {e1} {rel} {e2}."
            all_facts.append(fact)
    
    # Add final fact once
    final_fact = f"PROPERTY: {shared_entity}'s {final_prop} is {final_value}."
    all_facts.append(final_fact)
    
    # Add noise and build context
    random.shuffle(all_facts)
    
    fact_tokens = sum(len(f) // 4 for f in all_facts)
    noise_tokens = context_tokens - fact_tokens - 60
    
    positions = sorted([random.uniform(0.1, 0.9) for _ in range(len(all_facts))])
    
    context_parts = []
    prev_pos = 0.0
    
    for fact, pos in zip(all_facts, positions):
        noise_for_gap = int(noise_tokens * (pos - prev_pos))
        if noise_for_gap > 10:
            context_parts.append(generate_noise_text(noise_for_gap))
        context_parts.append(fact)
        prev_pos = pos
    
    context_parts.append(generate_noise_text(max(50, noise_tokens // 4)))
    context = " ".join(context_parts)
    
    # Generate query
    starts_str = " and ".join(chain_starts)
    query = f"Question: What is the {final_prop} of the entity that {starts_str} both lead to? Answer:"
    
    return {
        "context": context,
        "query": query,
        "answer": final_value,
        "metadata": {
            "n_chains": n_chains,
            "hops_per_chain": hops_per_chain,
            "shared_entity": shared_entity,
            "chain_starts": chain_starts,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate dependency tracing data")
    parser.add_argument("--context_len", type=int, default=32000, help="Context length in tokens")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data/dependency.jsonl", help="Output file")
    parser.add_argument("--hops", type=int, default=2, help="Number of hops in chain")
    parser.add_argument("--complex", action="store_true", help="Generate complex multi-chain samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_samples} samples with {args.hops} hops, context ~{args.context_len} tokens")
    
    with open(output_path, 'w') as f:
        for i in tqdm(range(args.num_samples)):
            if args.complex:
                sample = generate_complex_reasoning_sample(
                    context_tokens=args.context_len,
                    n_chains=random.randint(2, 3),
                    hops_per_chain=args.hops,
                )
            else:
                sample = generate_dependency_sample(
                    context_tokens=args.context_len,
                    n_hops=args.hops,
                    add_distractors=True,
                )
            
            training_sample = {
                "input": sample["context"] + " " + sample["query"],
                "target": sample["answer"],
                "metadata": sample["metadata"],
            }
            
            f.write(json.dumps(training_sample) + "\n")
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
