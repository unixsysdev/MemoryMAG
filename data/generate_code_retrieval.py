"""
Code Retrieval Data Generation for MemoryMAG Phase 3 Training.

Generates real-world code retrieval tasks from repositories:
- Parse repo to find function/class definitions
- Find usages that require distant definitions
- Create context: [definition] + [noise files] + [usage site]
- Target: completion requiring the distant definition

This teaches semantic retrieval in realistic coding scenarios.

Usage:
    python generate_code_retrieval.py --repo_path /path/to/repo --output data/code_retrieval.jsonl
"""

import argparse
import json
import random
import ast
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from tqdm import tqdm
import re


@dataclass
class CodeDefinition:
    """A code definition (function, class, constant)."""
    name: str
    type: str  # 'function', 'class', 'constant'
    code: str
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None


@dataclass
class CodeUsage:
    """A usage of a code definition."""
    definition: CodeDefinition
    usage_code: str
    file_path: str
    line_number: int
    context_before: str  # Code context before usage
    context_after: str   # Code context after usage


def parse_python_file(file_path: str) -> List[CodeDefinition]:
    """Parse a Python file to extract definitions."""
    definitions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function
                code = '\n'.join(lines[node.lineno - 1:node.end_lineno])
                docstring = ast.get_docstring(node)
                
                definitions.append(CodeDefinition(
                    name=node.name,
                    type='function',
                    code=code,
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                    docstring=docstring,
                ))
            
            elif isinstance(node, ast.ClassDef):
                # Extract class
                code = '\n'.join(lines[node.lineno - 1:node.end_lineno])
                docstring = ast.get_docstring(node)
                
                definitions.append(CodeDefinition(
                    name=node.name,
                    type='class',
                    code=code,
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                    docstring=docstring,
                ))
            
            elif isinstance(node, ast.Assign):
                # Extract top-level constants (UPPER_CASE)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        code = '\n'.join(lines[node.lineno - 1:node.end_lineno])
                        
                        definitions.append(CodeDefinition(
                            name=target.id,
                            type='constant',
                            code=code,
                            file_path=file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno,
                        ))
    
    except Exception as e:
        pass  # Skip files that can't be parsed
    
    return definitions


def find_usages_in_file(
    file_path: str, 
    definitions: List[CodeDefinition],
    context_lines: int = 10,
) -> List[CodeUsage]:
    """Find usages of definitions in a file."""
    usages = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        for defn in definitions:
            # Skip if definition is in the same file
            if defn.file_path == file_path:
                continue
            
            # Search for usage
            pattern = rf'\b{re.escape(defn.name)}\b'
            
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    # Found usage
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    
                    usages.append(CodeUsage(
                        definition=defn,
                        usage_code=line,
                        file_path=file_path,
                        line_number=i + 1,
                        context_before='\n'.join(lines[start:i]),
                        context_after='\n'.join(lines[i+1:end]),
                    ))
    
    except Exception as e:
        pass
    
    return usages


def scan_repository(repo_path: str) -> Tuple[List[CodeDefinition], List[CodeUsage]]:
    """Scan a repository for definitions and their usages."""
    all_definitions = []
    all_usages = []
    
    python_files = list(Path(repo_path).rglob('*.py'))
    
    print(f"Scanning {len(python_files)} Python files...")
    
    # First pass: collect all definitions
    for file_path in tqdm(python_files, desc="Parsing definitions"):
        definitions = parse_python_file(str(file_path))
        all_definitions.extend(definitions)
    
    print(f"Found {len(all_definitions)} definitions")
    
    # Second pass: find usages
    for file_path in tqdm(python_files, desc="Finding usages"):
        usages = find_usages_in_file(str(file_path), all_definitions)
        all_usages.extend(usages)
    
    print(f"Found {len(all_usages)} usages")
    
    return all_definitions, all_usages


def generate_noise_code(target_tokens: int) -> str:
    """Generate realistic-looking code noise."""
    templates = [
        # Function templates
        '''def process_{name}(data):
    """Process {name} data."""
    result = []
    for item in data:
        if item.{attr}:
            result.append(item.{method}())
    return result
''',
        # Class templates
        '''class {Name}Handler:
    """Handler for {name} operations."""
    
    def __init__(self, config):
        self.config = config
        self._{attr} = None
    
    def process(self, input_data):
        return self._{method}(input_data)
    
    def _{method}(self, data):
        return data
''',
        # Config templates
        '''{NAME}_CONFIG = {{
    "enabled": True,
    "timeout": {num},
    "retries": {num2},
    "endpoint": "{endpoint}",
}}
''',
        # Utility function templates
        '''def validate_{name}(value):
    if value is None:
        raise ValueError("{name} cannot be None")
    if not isinstance(value, {type}):
        raise TypeError(f"Expected {type}, got {{type(value)}}")
    return value
''',
    ]
    
    names = ["user", "data", "request", "response", "item", "record", "entry", "config", "state"]
    attrs = ["value", "status", "id", "name", "type", "data"]
    methods = ["process", "validate", "transform", "convert", "parse"]
    types = ["str", "int", "dict", "list", "bool"]
    
    target_chars = target_tokens * 4
    result = []
    current_chars = 0
    
    while current_chars < target_chars:
        template = random.choice(templates)
        name = random.choice(names)
        
        code = template.format(
            name=name,
            Name=name.capitalize(),
            NAME=name.upper(),
            attr=random.choice(attrs),
            method=random.choice(methods),
            type=random.choice(types),
            num=random.randint(10, 100),
            num2=random.randint(1, 5),
            endpoint=f"https://api.example.com/{name}",
        )
        
        result.append(code)
        current_chars += len(code)
    
    return '\n\n'.join(result)


def create_completion_task(usage: CodeUsage) -> Tuple[str, str, str]:
    """
    Create a code completion task from a usage.
    
    Returns:
        prompt: Code context with something to complete
        target: The completion (using the definition)
        definition_needed: The definition required to complete correctly
    """
    defn = usage.definition
    
    # Create prompt that requires knowing the definition
    if defn.type == 'constant':
        # For constants, mask the value
        prompt = f"{usage.context_before}\n{usage.usage_code.split(defn.name)[0]}"
        target = defn.name
    
    elif defn.type == 'function':
        # For functions, prompt for the function call
        prompt = f"{usage.context_before}\n# Use {defn.name} here\nresult = "
        target = f"{defn.name}("
    
    else:  # class
        # For classes, prompt for instantiation or method call
        prompt = f"{usage.context_before}\n# Create instance of {defn.name}\nobj = "
        target = f"{defn.name}("
    
    return prompt, target, defn.code


def generate_code_retrieval_sample(
    usage: CodeUsage,
    context_tokens: int,
    all_definitions: List[CodeDefinition],
) -> Dict:
    """
    Generate a code retrieval training sample.
    
    Args:
        usage: The code usage to base the sample on
        context_tokens: Target context length
        all_definitions: All available definitions for noise
        
    Returns:
        Training sample dict
    """
    defn = usage.definition
    
    # Create the completion task
    prompt, target, definition_code = create_completion_task(usage)
    
    # Calculate token budget
    definition_tokens = len(definition_code) // 4
    prompt_tokens = len(prompt) // 4
    noise_tokens = context_tokens - definition_tokens - prompt_tokens - 50
    
    # Generate context
    # Format: [definition] + [noise] + [prompt]
    
    # Add some distractor definitions
    distractors = []
    for d in random.sample(all_definitions, min(5, len(all_definitions))):
        if d.name != defn.name:
            distractors.append(f"# From {d.file_path}\n{d.code}")
    
    distractor_code = '\n\n'.join(distractors)
    distractor_tokens = len(distractor_code) // 4
    
    # Adjust noise
    noise_tokens = max(100, noise_tokens - distractor_tokens)
    noise_code = generate_noise_code(noise_tokens)
    
    # Build context
    context = f"""# File: {defn.file_path}
{definition_code}

# Other code in the repository...

{noise_code}

{distractor_code}

# File: {usage.file_path} (line {usage.line_number})
{prompt}"""
    
    return {
        "context": context,
        "prompt": prompt,
        "target": target,
        "definition": definition_code,
        "metadata": {
            "definition_name": defn.name,
            "definition_type": defn.type,
            "definition_file": defn.file_path,
            "usage_file": usage.file_path,
            "usage_line": usage.line_number,
        }
    }


def generate_synthetic_code_sample(
    context_tokens: int,
    task_type: str = "function_call",
) -> Dict:
    """
    Generate a synthetic code retrieval sample (no real repo needed).
    
    This is useful for initial training or when real repos aren't available.
    """
    
    if task_type == "constant":
        # Generate constant retrieval task
        const_name = f"{'_'.join(random.choices(['MAX', 'MIN', 'DEFAULT', 'CONFIG'], k=2))}_{random.choice(['SIZE', 'COUNT', 'TIMEOUT', 'LIMIT'])}"
        const_value = random.randint(1, 1000)
        
        definition = f"{const_name} = {const_value}"
        
        noise = generate_noise_code(context_tokens - 100)
        
        prompt = f"""# Using configuration
def process_batch(items):
    batch_size = """
        
        target = const_name
        
    elif task_type == "function_call":
        # Generate function call task
        func_name = f"{random.choice(['process', 'validate', 'transform', 'convert'])}_{random.choice(['data', 'input', 'request', 'item'])}"
        
        definition = f'''def {func_name}(data, strict=False):
    """Process the input data.
    
    Args:
        data: Input data to process
        strict: If True, raise on invalid data
        
    Returns:
        Processed data
    """
    if strict and not data:
        raise ValueError("Data cannot be empty")
    return {{"processed": data, "status": "ok"}}
'''
        
        noise = generate_noise_code(context_tokens - 200)
        
        prompt = f"""# Main processing logic
def handle_request(request):
    data = request.get("data")
    # Process the data using {func_name}
    result = """
        
        target = f"{func_name}(data"
        
    else:  # class
        class_name = f"{random.choice(['Data', 'Request', 'Response', 'Item'])}{random.choice(['Handler', 'Processor', 'Manager', 'Factory'])}"
        
        definition = f'''class {class_name}:
    """Handles {class_name.lower()} operations."""
    
    def __init__(self, config=None):
        self.config = config or {{}}
        self._initialized = True
    
    def process(self, data):
        """Process input data."""
        return {{"result": data, "handler": self.__class__.__name__}}
'''
        
        noise = generate_noise_code(context_tokens - 300)
        
        prompt = f"""# Initialize the handler
config = {{"timeout": 30}}
handler = """
        
        target = f"{class_name}(config"
    
    context = f"""# definitions.py
{definition}

# Other modules...
{noise}

# main.py
{prompt}"""
    
    return {
        "input": context,
        "target": target,
        "metadata": {
            "task_type": task_type,
            "synthetic": True,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate code retrieval training data")
    parser.add_argument("--repo_path", type=str, default=None, help="Path to repository")
    parser.add_argument("--output", type=str, default="data/code_retrieval.jsonl", help="Output file")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--context_len", type=int, default=64000, help="Context length in tokens")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic samples only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.synthetic or args.repo_path is None:
        print(f"Generating {args.num_samples} synthetic code retrieval samples...")
        
        with open(output_path, 'w') as f:
            for _ in tqdm(range(args.num_samples)):
                task_type = random.choice(["constant", "function_call", "class"])
                sample = generate_synthetic_code_sample(
                    context_tokens=args.context_len,
                    task_type=task_type,
                )
                f.write(json.dumps(sample) + "\n")
    
    else:
        print(f"Scanning repository: {args.repo_path}")
        definitions, usages = scan_repository(args.repo_path)
        
        if not usages:
            print("No usages found. Generating synthetic samples instead.")
            args.synthetic = True
            main()
            return
        
        print(f"Generating {args.num_samples} samples from {len(usages)} usages...")
        
        with open(output_path, 'w') as f:
            for _ in tqdm(range(args.num_samples)):
                usage = random.choice(usages)
                sample = generate_code_retrieval_sample(
                    usage=usage,
                    context_tokens=args.context_len,
                    all_definitions=definitions,
                )
                
                training_sample = {
                    "input": sample["context"],
                    "target": sample["target"],
                    "metadata": sample["metadata"],
                }
                
                f.write(json.dumps(training_sample) + "\n")
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
