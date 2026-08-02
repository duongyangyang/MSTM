"""
Split raw JSONL dataset into train/val/test with stratified category balance.

Usage:
    # Split all categories from data/raw/ into data/processed/
    python split_dataset.py

    # Custom paths
    python split_dataset.py --input data/raw/ --output data/processed/ --train 0.8 --val 0.1 --test 0.1

    # Dry run with stats only
    python split_dataset.py --dry-run

See TODOLIST.md Phase 2, Task 5.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


def load_all_examples(input_dir: Path) -> list[dict]:
    """Load all JSONL examples from a directory, one file per category."""
    examples = []
    jsonl_files = sorted(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {input_dir}")

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError:
                    print(f"  Warning: skipping unparseable line in {fpath.name}")
                    continue

    return examples


def validate_example(example: dict) -> bool:
    """Check required schema fields."""
    required = {"M", "delta_M", "M_prime", "category"}
    return all(
        k in example and isinstance(example[k], str) and len(example[k]) > 0
        for k in required
    )


def stratified_split(
    examples: list[dict],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split examples into train/val/test, maintaining category proportions.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 0.001

    rng = random.Random(seed)

    # Group by category
    by_category: dict[str, list[dict]] = {}
    for ex in examples:
        cat = ex.get("category", "unknown")
        by_category.setdefault(cat, []).append(ex)

    train, val, test = [], [], []

    for cat, items in by_category.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        # Adjust to exactly sum to n
        n_test = n - n_train - n_val

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    # Shuffle within each split to mix categories
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def save_jsonl(examples: list[dict], output_path: Path) -> None:
    """Save examples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def print_stats(examples: list[dict], label: str) -> None:
    """Print summary statistics for a split."""
    if not examples:
        print(f"  {label}: 0 examples")
        return

    cat_counts = Counter(ex["category"] for ex in examples)
    avg_m = sum(len(e["M"]) for e in examples) / len(examples)
    avg_d = sum(len(e["delta_M"]) for e in examples) / len(examples)
    avg_mp = sum(len(e["M_prime"]) for e in examples) / len(examples)
    compression = avg_mp / (avg_m + avg_d) if (avg_m + avg_d) > 0 else 0

    print(f"\n  {label}: {len(examples)} examples")
    print(f"    Categories: {dict(cat_counts)}")
    print(f"    Avg |M|: {avg_m:.0f}  |ΔM|: {avg_d:.0f}  |M′|: {avg_mp:.0f}")
    print(f"    Compression (M′/(M+ΔM)): {compression:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Split raw JSONL dataset into train/val/test"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/",
        help="Directory containing category JSONL files (default: data/raw/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Output directory for split files (default: data/processed/)",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Training fraction (default: 0.8)",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Validation fraction (default: 0.1)",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test fraction (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only, don't save files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Load
    print(f"Loading examples from {input_dir}...")
    examples = load_all_examples(input_dir)
    print(f"  Loaded {len(examples)} total examples")

    # Validate
    valid = [e for e in examples if validate_example(e)]
    invalid = len(examples) - len(valid)
    if invalid > 0:
        print(f"  ⚠️  {invalid} examples failed validation, removed")

    if not valid:
        print("ERROR: No valid examples found.")
        return

    # Category distribution (before split)
    cat_counts = Counter(e["category"] for e in valid)
    print(f"\n  Category distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}")

    # Split
    print(f"\nSplitting (train={args.train:.0%}, val={args.val:.0%}, test={args.test:.0%})...")
    train, val, test = stratified_split(
        valid,
        train_frac=args.train,
        val_frac=args.val,
        test_frac=args.test,
        seed=args.seed,
    )

    # Print stats
    print_stats(train, "TRAIN")
    print_stats(val, "VAL")
    print_stats(test, "TEST")

    total_split = len(train) + len(val) + len(test)
    print(f"\n  Total across splits: {total_split} (original: {len(valid)})")

    if args.dry_run:
        print("\n[Dry run — no files saved]")
        return

    # Save combined splits
    print(f"\nSaving to {output_dir}...")
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "val.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")
    print(f"  Saved: train.jsonl, val.jsonl, test.jsonl")

    # Also save per-category files for convenience (useful for ablation analysis)
    for label, split in [("train", train), ("val", val), ("test", test)]:
        by_cat: dict[str, list] = {}
        for ex in split:
            by_cat.setdefault(ex["category"], []).append(ex)
        for cat, items in by_cat.items():
            cat_path = output_dir / label / f"{cat}.jsonl"
            save_jsonl(items, cat_path)
    print(f"  Saved: per-category splits under train/ val/ test/")

    # Save a split summary
    summary = {
        "input_dir": str(input_dir),
        "total_loaded": len(examples),
        "total_valid": len(valid),
        "split_ratios": {"train": args.train, "val": args.val, "test": args.test},
        "seed": args.seed,
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "category_distribution": {
            "overall": dict(cat_counts),
            "train": dict(Counter(e["category"] for e in train)),
            "val": dict(Counter(e["category"] for e in val)),
            "test": dict(Counter(e["category"] for e in test)),
        },
    }
    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: split_summary.json")

    print("\nDone!")


if __name__ == "__main__":
    main()