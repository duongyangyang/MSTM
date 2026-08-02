#!/bin/bash
# Parallel dataset generation — all 5 categories simultaneously
# ================================================================
# Each category runs as a separate process, iterating through all
# 10 domains and appending to the existing data/raw/{category}.jsonl.
#
# Usage:
#   # Generate 100 examples per domain for each category (1000 per category)
#   bash scripts/parallel_generate.sh
#
#   # Generate 200 per domain, more examples
#   bash scripts/parallel_generate.sh --n 200
#
#   # Custom categories and domains
#   bash scripts/parallel_generate.sh --categories update,consolidation --domains health,career,finance
#
#   # Use a specific model
#   bash scripts/parallel_generate.sh --model gpt-4o-mini
#
# Requirements:
#   - OPENAI_API_KEY env var (or --api-key)
# ================================================================

set -e

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CATEGORIES="update,contradiction_resolution,consolidation,abstraction,forgetting"
DOMAINS="health,finance,career,education,relationships,travel,hobbies,beliefs_values,technology,daily_life"
N_PER_DOMAIN=100
MODEL="gpt-4o"
OUT_DIR="data/raw"
API_KEY="${OPENAI_API_KEY:-}"
BASE_URL="${OPENAI_BASE_URL:-}"
BATCH_SIZE=10
TEMP=0.8

while [[ $# -gt 0 ]]; do
    case $1 in
        --categories) CATEGORIES="$2"; shift 2 ;;
        --domains) DOMAINS="$2"; shift 2 ;;
        --n) N_PER_DOMAIN="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --temperature) TEMP="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash scripts/parallel_generate.sh [OPTIONS]"
            echo ""
            echo "Launches all 5 categories in parallel. Each writes to data/raw/{category}.jsonl"
            echo ""
            echo "Options:"
            echo "  --categories CATS   Comma-separated categories (default: all 5)"
            echo "  --domains DOMS      Comma-separated domains (default: all 10)"
            echo "  --n N               Examples per domain per category (default: 100)"
            echo "  --model MODEL       LLM model (default: gpt-4o)"
            echo "  --api-key KEY       API key (default: \$OPENAI_API_KEY)"
            echo "  --base-url URL      API base URL (default: \$OPENAI_BASE_URL)"
            echo "  --out DIR           Output directory (default: data/raw)"
            echo "  --batch-size N      Examples per API call (default: 10)"
            echo "  --temperature T     Sampling temperature (default: 0.8)"
            echo ""
            echo "Examples:"
            echo "  bash scripts/parallel_generate.sh"
            echo "  bash scripts/parallel_generate.sh --n 200"
            echo "  bash scripts/parallel_generate.sh --categories update,consolidation --domains health,career"
            echo "  bash scripts/parallel_generate.sh --model gpt-4o-mini --n 150"
            exit 0 ;;
        *) echo "Unknown: $1. Use --help."; exit 1 ;;
    esac
done

cd "$PROJECT_DIR"

IFS=',' read -ra CAT_ARRAY <<< "$CATEGORIES"
IFS=',' read -ra DOM_ARRAY <<< "$DOMAINS"

TOTAL_DOMS=${#DOM_ARRAY[@]}
TOTAL_PER_CAT=$((N_PER_DOMAIN * TOTAL_DOMS))

echo "========================================"
echo " Parallel Dataset Generation"
echo "========================================"
echo " Categories: ${#CAT_ARRAY[@]} (${CATEGORIES})"
echo " Domains:   ${TOTAL_DOMS} (${DOMAINS})"
echo " Per domain: ${N_PER_DOMAIN} examples"
echo " Per category total: ${TOTAL_PER_CAT}"
echo " Model:     ${MODEL}"
echo " Output:    ${OUT_DIR}/"
echo "========================================"
echo ""

PY_ARGS=""
[ -n "$API_KEY" ] && PY_ARGS="$PY_ARGS --api-key $API_KEY"
[ -n "$BASE_URL" ] && PY_ARGS="$PY_ARGS --base-url $BASE_URL"

PIDS=()
FAILED=0

for cat in "${CAT_ARRAY[@]}"; do
    LOGFILE="/tmp/mstm_gen_${cat}.log"
    echo "  [START] ${cat} → ${OUT_DIR}/${cat}.jsonl (log: ${LOGFILE})"

    python3 src/data_generation/generate.py \
        --category "$cat" \
        --domain all \
        --n "$N_PER_DOMAIN" \
        --out "$OUT_DIR" \
        --model "$MODEL" \
        --batch-size "$BATCH_SIZE" \
        --temperature "$TEMP" \
        --continue \
        $PY_ARGS \
        > "$LOGFILE" 2>&1 &

    PIDS+=("$!")
done

echo ""
echo "All ${#PIDS[@]} categories running in parallel..."
echo ""

# Wait for all
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    cat="${CAT_ARRAY[$i]}"
    wait "$pid" 2>/dev/null
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "  [DONE]  ${cat}"
    else
        echo "  [FAIL]  ${cat} (exit code: ${EXIT_CODE}) — see /tmp/mstm_gen_${cat}.log"
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo " Generation Complete!"
echo "========================================"

python3 -c "
import json
from collections import Counter
from pathlib import Path

out_dir = Path('${OUT_DIR}')
cat_counts = Counter()
total = 0

for f in sorted(out_dir.glob('*.jsonl')):
    count = 0
    with open(f) as fh:
        for line in fh:
            if line.strip():
                try:
                    obj = json.loads(line)
                    cat_counts[obj.get('category','?')] += 1
                    count += 1
                    total += 1
                except: pass
    if count > 0:
        print(f'  {f.name}: {count}')

print(f'\n  TOTAL: {total}')
print(f'\n=== By Category ===')
for k, v in cat_counts.most_common():
    print(f'  {k}: {v}')
"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  ${FAILED} category(ies) failed. Check logs in /tmp/mstm_gen_*.log"
fi

echo ""
echo "Next: python src/data_generation/split_dataset.py"