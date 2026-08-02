"""
Dataset generation entry point.

Generates (M, delta_M, M_prime) triplets by prompting a strong LLM (GPT-4o),
using the category-specific templates in prompts/.

Usage:
    # Pilot: 100 examples per category
    python generate.py --category update --n 100 --out ../../data/raw/update_pilot.jsonl

    # Full: 1000 examples per category (total 5000 across 5 categories)
    python generate.py --category all --n 1000 --out ../../data/raw/

See TODOLIST.md Phase 2, Task 4-5.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


CATEGORIES = [
    "update",
    "contradiction_resolution",
    "consolidation",
    "abstraction",
    "forgetting",
]

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(category: str) -> str:
    """Load the prompt template for a given category."""
    prompt_path = PROMPT_DIR / f"{category}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def get_openai_client(api_key: str = None, base_url: str = None):
    """Initialize and return an OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        )

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not provided. Set --api-key or OPENAI_API_KEY env var."
        )

    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def generate_batch(
    client,
    prompt: str,
    n: int,
    model: str = "gpt-4o",
    temperature: float = 0.8,
    max_tokens: int = 4096,
    batch_size: int = 10,
    verbose: bool = True,
    output_path: str = None,
    append: bool = False,
) -> list[dict]:
    """
    Generate n examples by calling the LLM repeatedly.

    The prompt template instructs the model to output N examples as JSONL.
    We request batch_size examples per call and aggregate.

    If output_path is provided, each batch is saved to disk immediately
    (incremental save — protects against crashes mid-generation).
    """
    all_examples = []
    remaining = n
    consecutive_empty = 0
    max_empty_retries = 3

    # Determine file mode for incremental saves
    if output_path:
        file_mode = "a" if append else "w"
        # Clear file on first write if not appending
        if not append:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            open(output_path, "w").close()

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        batch_prompt = prompt.replace(
            "Generate 100 diverse examples",
            f"Generate {current_batch} diverse examples",
        )

        if verbose:
            print(
                f"  Requesting {current_batch} examples... "
                f"({len(all_examples)}/{n} collected so far)"
            )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data generation assistant. Output ONLY valid JSONL "
                            "(one JSON object per line), no markdown fences, no commentary. "
                            "Each line must be a complete, parseable JSON object."
                        ),
                    },
                    {"role": "user", "content": batch_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            raw_output = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            clean_output = raw_output
            if clean_output.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = clean_output.find("\n")
                if first_newline != -1:
                    clean_output = clean_output[first_newline + 1:]
                # Remove closing fence
                if clean_output.rstrip().endswith("```"):
                    clean_output = clean_output.rstrip()[:-3].strip()

            # Parse — try JSONL first, then single JSON object, then JSON array
            batch_examples = []

            # Attempt 1: JSONL (line-by-line)
            for line in clean_output.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("```"):
                    continue
                try:
                    obj = json.loads(line)
                    batch_examples.append(obj)
                except json.JSONDecodeError:
                    continue

            # Attempt 2: single JSON object (LLM pretty-printed it)
            if len(batch_examples) == 0:
                try:
                    obj = json.loads(clean_output)
                    if isinstance(obj, dict):
                        batch_examples = [obj]
                    elif isinstance(obj, list):
                        batch_examples = obj
                except json.JSONDecodeError:
                    pass

            if len(batch_examples) == 0:
                consecutive_empty += 1
                if verbose:
                    print(f"    Got 0 parseable JSON lines (consecutive: {consecutive_empty}/{max_empty_retries})")
                    print(f"    Raw output preview: {raw_output[:200]}...")
                if consecutive_empty >= max_empty_retries:
                    print(f"  ⚠️  {max_empty_retries} consecutive empty batches — stopping. "
                          f"Collected {len(all_examples)}/{n} examples.")
                    break
                time.sleep(2)
                continue
            else:
                consecutive_empty = 0

            all_examples.extend(batch_examples)
            remaining = n - len(all_examples)

            # Incremental save: write batch to disk immediately
            if output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    for ex in batch_examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            if verbose:
                print(f"    Got {len(batch_examples)} parseable lines from this batch")

        except Exception as e:
            print(f"  Error during API call: {e}")
            print("  Retrying in 5 seconds...")
            time.sleep(5)
            continue

        # Rate limiting pause between batches
        if remaining > 0:
            time.sleep(1)

    return all_examples[:n]


def validate_example(example: dict, category: str) -> bool:
    """Validate that an example has the required schema."""
    required_keys = {"M", "delta_M", "M_prime", "category"}
    if not all(k in example for k in required_keys):
        return False
    if example["category"] != category:
        return False
    if not all(
        isinstance(example[k], str) and len(example[k]) > 0
        for k in ["M", "delta_M", "M_prime"]
    ):
        return False
    return True


def save_examples(examples: list[dict], output_path: str, append: bool = False) -> None:
    """Save examples to a JSONL file."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    action = "Appended" if append else "Saved"
    print(f"{action} {len(examples)} examples to {out_path}")


def count_existing(file_path: str) -> int:
    """Count existing valid JSON lines in a JSONL file."""
    path = Path(file_path)
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue
    return count


def print_stats(examples: list[dict], category: str) -> None:
    """Print summary statistics for a batch of examples."""
    if not examples:
        print("  No examples to summarize.")
        return

    avg_m_len = sum(len(e["M"]) for e in examples) / len(examples)
    avg_delta_len = sum(len(e["delta_M"]) for e in examples) / len(examples)
    avg_mprime_len = sum(len(e["M_prime"]) for e in examples) / len(examples)

    print(f"\n  Category: {category}")
    print(f"  Total examples: {len(examples)}")
    print(f"  Avg |M|: {avg_m_len:.0f} chars")
    print(f"  Avg |ΔM|: {avg_delta_len:.0f} chars")
    print(f"  Avg |M′|: {avg_mprime_len:.0f} chars")


def main():
    parser = argparse.ArgumentParser(
        description="Generate (M, delta_M, M_prime) triplets for memory state transition"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help=f"Category to generate, or 'all'. Choices: {CATEGORIES + ['all']}",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of examples per category (default: 100)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory or file path for JSONL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL for third-party providers (default: OPENAI_BASE_URL env var or OpenAI default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Examples per API call (default: 10)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate examples against schema (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip schema validation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_gen",
        help="Append to existing file instead of overwriting (generates only up to --n total)",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Determine categories to generate
    if args.category == "all":
        categories = CATEGORIES
    elif args.category in CATEGORIES:
        categories = [args.category]
    else:
        print(f"Error: unknown category '{args.category}'")
        print(f"Valid categories: {CATEGORIES + ['all']}")
        sys.exit(1)

    # Initialize client
    base_url_info = f", base_url: {args.base_url}" if args.base_url else ""
    print(f"Initializing client (model: {args.model}{base_url_info})...")
    client = get_openai_client(api_key=args.api_key, base_url=args.base_url)

    # Determine output path
    out_path = Path(args.out)
    if len(categories) > 1 or out_path.suffix != ".jsonl":
        # Multi-category: treat --out as directory
        out_path.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    all_validated = []

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Generating category: {category}")
        print(f"{'='*60}")

        # Determine file path for this category
        if len(categories) > 1 or out_path.suffix != ".jsonl":
            file_path = out_path / f"{category}.jsonl"
        else:
            file_path = out_path

        # Handle --continue: count existing, adjust target
        existing_count = 0
        if args.continue_gen:
            existing_count = count_existing(str(file_path))
            if existing_count >= args.n:
                print(f"  Already have {existing_count} examples (target: {args.n}) — skipping")
                all_validated.append((category, existing_count))
                total_examples += existing_count
                continue
            print(f"  Found {existing_count} existing examples, generating {args.n - existing_count} more")

        # Load prompt
        prompt = load_prompt(category)
        if verbose:
            print(f"  Loaded prompt ({len(prompt)} chars)")

        # Generate examples
        n_to_generate = args.n - existing_count if args.continue_gen else args.n
        examples = generate_batch(
            client,
            prompt,
            n=n_to_generate,
            model=args.model,
            temperature=args.temperature,
            batch_size=args.batch_size,
            verbose=verbose,
            output_path=str(file_path),
            append=args.continue_gen,
        )

        # Validate
        if args.validate:
            valid = [e for e in examples if validate_example(e, category)]
            invalid_count = len(examples) - len(valid)
            if invalid_count > 0:
                print(f"  ⚠️  {invalid_count} examples failed validation, removed")
            examples = valid
        else:
            # Still assign category if missing
            for e in examples:
                if "category" not in e:
                    e["category"] = category

        # Print stats
        print_stats(examples, category)

        # Final save: overwrite with validated-only data
        # For --continue, preserve existing validated data from previous runs
        if args.continue_gen:
            existing = []
            if Path(file_path).exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                existing.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            all_examples_final = existing + examples
        else:
            all_examples_final = examples

        save_examples(all_examples_final, str(file_path), append=False)
        total_examples += len(examples)
        all_validated.append((category, len(all_examples_final)))

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"{'='*60}")
    for cat, count in all_validated:
        print(f"  {cat}: {count} examples")
    print(f"  Total: {total_examples} examples")


if __name__ == "__main__":
    main()