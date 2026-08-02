# GPUCLOUD.md — Cloud GPU Operations Guide

## Strategy: GPU Inference on Cloud, Eval on Local

```
Cloud (4090)                    Local (Mac)
────────────                    ──────────
Train MSTM (~5 min)             tar -xzf
MSTM inference → save M′        Load M′ from JSONL
Save baseline memories          GPT-4o-mini answer generation
Package everything              Compute metrics (F1, BLEU, ROUGE...)
                                No GPU needed
```

This avoids paying for GPU idle time while waiting for GPT-4o-mini API responses.

---

## Part 1: Cloud GPU — Train + Save Memory

### 1. Upload project to cloud

```bash
# On your Mac
cd ~/Downloads/memory-state-transition
tar -czf /tmp/mstm_project.tar.gz \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'papers' --exclude 'data/raw' \
    src/ configs/ scripts/ data/processed/ requirements.txt

scp /tmp/mstm_project.tar.gz user@your-gpu-instance:~/

# On cloud
tar -xzf mstm_project.tar.gz && cd memory-state-transition
```

### 2. Run

```bash
bash scripts/cloud_train.sh
```

This does:
1. Check GPU
2. Install deps
3. Train MSTM (Qwen3-0.6B, LoRA) — ~5 min
4. MSTM inference on LoCoMo + LongMemEval — save M′ to JSONL — ~15 min
5. Save baseline memories (static, time_decay, heuristic, locomo_full) — ~2 min
6. Package everything into one `.tar.gz`

**Output:** `mstm_full_YYYYMMDD_HHMMSS.tar.gz` (~10 MB checkpoint + ~50 MB memory files)

### 3. Download

```bash
# On your Mac
scp user@your-gpu-instance:~/memory-state-transition/mstm_full_*.tar.gz ./
tar -xzf mstm_full_*.tar.gz
```

You now have:
```
experiments/
├── checkpoint/mstm_4090/     # Trained LoRA weights
├── memory/
│   ├── mstm_locomo.jsonl         # MSTM M′ for LoCoMo
│   ├── mstm_longmemeval.jsonl    # MSTM M′ for LongMemEval
│   ├── static_memory_locomo.jsonl
│   ├── time_decay_locomo.jsonl
│   ├── heuristic_consolidation_locomo.jsonl
│   └── locomo_full_locomo.jsonl
└── logs/
```

### 4. Shut down cloud instance

---

## Part 2: Local — Evaluate

### 2a. GPT-4o-mini answer generation

```bash
export OPENAI_API_KEY="sk-..."

# MSTM on LoCoMo
python src/eval/run_eval.py \
    --method mstm \
    --load-memory experiments/memory/mstm_locomo.jsonl \
    --benchmark locomo \
    --foundation-models gpt-4o-mini \
    --top-k 10 \
    --out results/

# All baselines on LoCoMo
for mem in experiments/memory/*_locomo.jsonl; do
    method=$(basename "$mem" _locomo.jsonl)
    python src/eval/run_eval.py \
        --method "$method" \
        --load-memory "$mem" \
        --benchmark locomo \
        --foundation-models gpt-4o-mini \
        --top-k 10 \
        --out results/
done
```

### 2b. Internal transition eval (Track 2 — no API needed)

```bash
python src/eval/run_eval.py \
    --mode transition \
    --method mstm \
    --checkpoint experiments/checkpoint/mstm_4090 \
    --test_data data/processed/test.jsonl \
    --out results/
```

### 2c. k-sensitivity

```bash
python src/eval/run_eval.py \
    --method mstm \
    --load-memory experiments/memory/mstm_locomo.jsonl \
    --benchmark locomo \
    --k-sweep 10,20,30,40,50 \
    --foundation-models gpt-4o-mini \
    --out results/
```

### 2d. Smoke test

```bash
python src/eval/run_eval.py \
    --method mstm \
    --load-memory experiments/memory/mstm_locomo.jsonl \
    --benchmark locomo \
    --foundation-models gpt-4o-mini \
    --max-samples 10 \
    --out results/
```

---

## Quick Reference

| Task | Where | Command |
|------|-------|---------|
| Train + save memory | Cloud | `bash scripts/cloud_train.sh` |
| Download | Local | `scp user@host:mstm_full_*.tar.gz ./` |
| Eval with GPT-4o-mini | Local | `--load-memory <file> --foundation-models gpt-4o-mini` |
| Transition eval | Local | `--mode transition --checkpoint <path>` |
| k-sweep | Local | `--k-sweep 10,20,30,40,50` |
| Smoke test | Local | `--max-samples 10` |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `batch_size` in config to 4 |
| `split.jsonl` not found | Run `python src/data_generation/split_dataset.py` first |
| memory JSONL too large | LoCoMo ~40 MB, LongMemEval ~5 MB. Fine for scp. |
| `--load-memory` not working | File must be JSONL with `question`, `ground_truth`, `category`, `memory` fields |