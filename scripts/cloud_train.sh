#!/bin/bash
# MSTM: Train + Save Memory — Cloud GPU
# ============================================
# 1. Train MSTM on cloud GPU
# 2. Run MSTM inference to save M' for all conversations
# 3. Package checkpoint + memory → download to local
# 4. Locally: load memory → GPT answer → eval (no GPU needed)
#
# Usage:
#   1. Upload repo to cloud GPU instance
#   2. bash scripts/cloud_train.sh
#   3. Download mstm_full_*.tar.gz
#   4. Run eval locally (see GPUCLOUD.md)
#
# Requirements: RTX 4090 (24GB+) or equivalent
# ============================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_ARCHIVE="mstm_full_${TIMESTAMP}.tar.gz"

cd "$PROJECT_DIR"

echo "========================================"
echo " MSTM: Train + Save Memory"
echo " Started: $(date)"
echo "========================================"

# ── Step 1: Check GPU ──────────────────────────────────────────────────

echo ""
echo "[1/5] Checking GPU..."

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'  PyTorch: {torch.__version__}')
"

# ── Step 2: Install dependencies ───────────────────────────────────────

echo ""
echo "[2/5] Installing dependencies..."

pip install --quiet torch transformers datasets peft accelerate \
    sentence-transformers scikit-learn openai huggingface_hub \
    pyyaml nltk numpy tiktoken

echo "  Done."

# ── Step 3: Verify data ────────────────────────────────────────────────

echo ""
echo "[3/5] Verifying data..."

python -c "
import json
from pathlib import Path
for split in ['train', 'val', 'test']:
    p = Path(f'data/processed/{split}.jsonl')
    if p.exists():
        lines = len([l for l in open(p) if l.strip()])
        print(f'  {split}: {lines} examples')
    else:
        print(f'  ⚠️  {split}.jsonl not found!')
        exit(1)
"

# ── Step 4: Train MSTM ─────────────────────────────────────────────────

echo ""
echo "[4/5] Training MSTM (Qwen3-0.6B, LoRA, SFT)..."
echo ""

python src/mstm/train.py --config configs/mstm_sft_4090.yaml \
    --output experiments/checkpoints/mstm_4090

echo ""
echo "  Training complete."

# ── Step 5: Save MSTM memory (GPU inference) ───────────────────────────

echo ""
echo "[5/5] Saving MSTM memory states for all benchmarks..."
echo "  (This runs MSTM inference on GPU, saves M' to JSONL)"
echo ""

# Save memory for LoCoMo
python src/eval/run_eval.py \
    --method mstm \
    --checkpoint experiments/checkpoints/mstm_4090 \
    --benchmark locomo \
    --save-memory experiments/memory/mstm_locomo.jsonl \
    --out /tmp/dummy

# Save memory for LongMemEval
python src/eval/run_eval.py \
    --method mstm \
    --checkpoint experiments/checkpoints/mstm_4090 \
    --benchmark longmemeval \
    --save-memory experiments/memory/mstm_longmemeval.jsonl \
    --out /tmp/dummy

# Also save baseline memories (fast, no GPU needed)
echo ""
echo "  Saving baseline memories..."
for method in static_memory time_decay heuristic_consolidation locomo_full; do
    echo "    $method..."
    python src/eval/run_eval.py \
        --method "$method" \
        --benchmark locomo \
        --save-memory "experiments/memory/${method}_locomo.jsonl" \
        --out /tmp/dummy
done

# ── Package ────────────────────────────────────────────────────────────

echo ""
echo "  Packaging..."

tar -czf "$OUTPUT_ARCHIVE" \
    experiments/checkpoints/mstm_4090 \
    experiments/memory/ \
    experiments/logs/ \
    configs/mstm_sft_4090.yaml \
    2>/dev/null

ARCHIVE_SIZE=$(du -h "$OUTPUT_ARCHIVE" | cut -f1)

echo ""
echo "========================================"
echo " Done!"
echo "  Archive: $OUTPUT_ARCHIVE ($ARCHIVE_SIZE)"
echo ""
echo "  Download to local:"
echo "    scp user@host:\$(pwd)/$OUTPUT_ARCHIVE ./"
echo ""
echo "  Then eval locally:"
echo "    tar -xzf $OUTPUT_ARCHIVE"
echo "    python src/eval/run_eval.py --method mstm --load-memory experiments/memory/mstm_locomo.jsonl --benchmark locomo --foundation-models gpt-4o-mini --out results/"
echo "========================================"