"""
SFT training script for MSTM.

Trains on (M, delta_M, M_prime) triplets from data/processed/.
Logs training cost (GPU-hours, examples seen) — required for RQ4 / TABLE.md Table 2.

Usage:
    # Train on pilot dataset
    python train.py --train_data ../../data/processed/train.jsonl \
                    --val_data ../../data/processed/val.jsonl \
                    --output ../../experiments/checkpoints/mstm_pilot \
                    --epochs 3 --batch_size 4

    # Train on full dataset with config file
    python train.py --config ../../configs/mstm_sft.yaml

See TODOLIST.md Phase 4, Tasks 9-10.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.mstm.model import MSTMModel, format_transition_prompt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MemoryTransitionDataset(Dataset):
    """
    PyTorch Dataset for (M, delta_M, M_prime) triplets.

    Each item is tokenized into the training prompt format:
    [system] + [M] + [delta_M] + [M_prime]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Load data
        self.examples = self._load_data(data_path)
        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def _load_data(self, path: str) -> List[Dict]:
        """Load JSONL data."""
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Validate required fields
                if all(k in obj for k in ["M", "delta_M", "M_prime"]):
                    examples.append(obj)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]

        # Build the full training prompt (with M_prime for teacher forcing)
        full_prompt = format_transition_prompt(
            M=example["M"],
            delta_M=example["delta_M"],
            M_prime=example["M_prime"],
            for_training=True,
        )

        # Build the prompt-only part (without M_prime) to know where to mask
        prompt_only = format_transition_prompt(
            M=example["M"],
            delta_M=example["delta_M"],
            M_prime="",
            for_training=False,
        )

        # Tokenize both (no padding for prompt_length calculation)
        full_tokens = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors="pt",
        )
        prompt_tokens = self.tokenizer(
            prompt_only,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_tokens["input_ids"].squeeze(0).tolist()
        attention_mask = full_tokens["attention_mask"].squeeze(0).tolist()

        # Create labels: mask the prompt part (set to -100) so the model
        # only learns to generate M_prime, not the fixed prompt format
        prompt_len = prompt_tokens["input_ids"].shape[1]
        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    model: MSTMModel,
    train_dataset: MemoryTransitionDataset,
    val_dataset: Optional[MemoryTransitionDataset] = None,
    output_dir: str = "./checkpoints",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 4,
    weight_decay: float = 0.01,
    logging_steps: int = 10,
    eval_steps: int = 100,
    save_steps: int = 500,
    max_grad_norm: float = 1.0,
    fp16: bool = True,
) -> Dict:
    """
    Train the MSTM model via SFT.

    Args:
        model: MSTMModel instance.
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
        output_dir: Directory to save checkpoints.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        learning_rate: Peak learning rate.
        gradient_accumulation_steps: Gradient accumulation steps.
        warmup_ratio: Fraction of steps for learning rate warmup.
        weight_decay: Weight decay for AdamW.
        logging_steps: Log every N steps.
        eval_steps: Evaluate every N steps.
        save_steps: Save checkpoint every N steps.
        max_grad_norm: Maximum gradient norm for clipping.
        fp16: Use mixed-precision training.

    Returns:
        Dict with training summary (loss curve, GPU hours, etc.).
    """
    from transformers import (
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=int(0.1 * epochs * len(train_dataset) / (batch_size * gradient_accumulation_steps)),
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        fp16=fp16 and model.device == "cuda",
        max_grad_norm=max_grad_norm,
        report_to="none",  # Disable wandb/tensorboard for simplicity
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=model.tokenizer,
        mlm=False,
    )

    # Track training start
    train_start = time.time()

    # Initialize trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print(f"\nStarting training...")
    print(f"  Device: {model.device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} x {gradient_accumulation_steps} accumulation")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training examples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Validation examples: {len(val_dataset)}")

    train_result = trainer.train()

    train_end = time.time()
    wall_hours = (train_end - train_start) / 3600

    # Save final model
    trainer.save_model(str(output_path))
    model.tokenizer.save_pretrained(str(output_path))
    print(f"\nModel saved to {output_path}")

    # Compute metrics
    metrics = train_result.metrics
    metrics["wall_clock_hours"] = wall_hours
    metrics["gpu_hours"] = wall_hours  # Simplified: assumes 1 GPU
    metrics["num_training_examples"] = len(train_dataset)
    metrics["train_samples_per_second"] = (
        len(train_dataset) * epochs / (wall_hours * 3600) if wall_hours > 0 else 0
    )

    # Save training metrics
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                **metrics,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "warmup_ratio": warmup_ratio,
                    "weight_decay": weight_decay,
                    "backbone": model.backbone_name,
                    "train_examples": len(train_dataset),
                    "val_examples": len(val_dataset) if val_dataset else 0,
                },
            },
            f,
            indent=2,
        )

    print(f"\nTraining complete!")
    print(f"  Wall time: {wall_hours:.2f} hours")
    print(f"  GPU hours: {metrics['gpu_hours']:.2f}")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Metrics saved to: {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SFT-train the Memory State Transition Model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI args)",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="../../data/processed/train.jsonl",
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="../../data/processed/val.jsonl",
        help="Path to validation data (JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../experiments/checkpoints/mstm",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model backbone",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--no_lora",
        action="store_false",
        dest="use_lora",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed-precision training",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_false",
        dest="fp16",
        help="Disable mixed-precision",
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Override args with config values
        if config.get("model", {}).get("backbone"):
            args.backbone = config["model"]["backbone"]
        if config.get("model", {}).get("max_seq_len"):
            args.max_seq_length = config["model"]["max_seq_len"]
        if config.get("training", {}).get("learning_rate"):
            args.learning_rate = config["training"]["learning_rate"]
        if config.get("training", {}).get("epochs"):
            args.epochs = config["training"]["epochs"]
        if config.get("training", {}).get("batch_size"):
            args.batch_size = config["training"]["batch_size"]
        if config.get("training", {}).get("lora") is not None:
            args.use_lora = config["training"]["lora"]
        if config.get("data", {}).get("train_path"):
            args.train_data = config["data"]["train_path"]
        if config.get("data", {}).get("val_path"):
            args.val_data = config["data"]["val_path"]

    # Initialize model
    model = MSTMModel(
        backbone=args.backbone,
        max_seq_length=args.max_seq_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Load datasets
    train_dataset = MemoryTransitionDataset(
        data_path=args.train_data,
        tokenizer=model.tokenizer,
        max_seq_length=args.max_seq_length,
    )

    val_dataset = None
    val_path = Path(args.val_data)
    if val_path.exists():
        val_dataset = MemoryTransitionDataset(
            data_path=args.val_data,
            tokenizer=model.tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        print(f"Validation data not found at {args.val_data}, skipping validation.")

    # Train
    metrics = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
    )

    return metrics


if __name__ == "__main__":
    main()