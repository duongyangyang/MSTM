# PROGRESS.md

Living status tracker. Update this at the end of each day (or each sync point). This is a status log, not a plan — plan lives in `WORKPLAN.md` and `TODOLIST.md`.

**How to update:** change status icons, fill in the "Notes / Blockers" cell, and add a dated entry to the Log at the bottom. Don't rewrite history — append.

**Status key:** ⬜ not started · 🔄 in progress · ✅ done · ⚠️ blocked · ❌ cut from scope

---

## Current status snapshot

*Update this line every sync point:*

**Day:** `4 / 14` **Current phase:** `Phase 3 (A-MEM pipeline refactor — retrieval, 6 metrics, foundation model sweep)` **Overall risk:** 🟢 on track

---

## Scope decisions (fill in as they're made — see WORKPLAN.md sync points)

| Decision | Status | Date decided | Notes |
|---|---|---|---|
| Primary SLM backbone | ✅ | 2026-08-01 | Qwen 3 0.6B (Qwen/Qwen3-0.6B) — updated from Qwen2.5-0.5B Aug 2 |
| Dataset target size | ✅ | 2026-08-01 | ~5,000 examples (1,000 per category) |
| Dataset generator LLM | ✅ | 2026-08-01 | GPT-4o |
| AgeMem: reproduce or reported-only | ⬜ | | Decide by Day 6 latest per WORKPLAN.md |
| A-MEM: reproduce or reported-only | ⬜ | | |

---

## Task status (mirrors TODOLIST.md numbering)

| # | Task | Track | Status | Notes / Blockers |
|---|---|---|---|---|
| 1 | Lock proposal & project parameters | Human | ✅ | SLM=Qwen2.5-0.5B, dataset=~5k, generator=GPT-4o |
| 2 | Deep-read AgeMem / LightMem / A-MEM | AI | ✅ | PDFs downloaded, summaries in papers/README.md, baseline READMEs updated |
| 3 | Draft Related Work section | AI | ✅ | Included in papers/README.md positioning matrix |
| 4 | Design & pilot triplet dataset | AI → Human review | ✅ | 5 prompt templates written (update, contradiction, consolidation, abstraction, forgetting) |
| 5 | Scale up & finalize dataset | AI → Human spot-check | 🔄 | Prompt templates fixed for compact M′; split_dataset.py ready; human regenerating abstraction + contradiction |
| 6 | Build evaluation harness | AI | ✅ | metrics.py upgraded with ROUGE-L, per-category, transition-quality, MQ, LLM-consistency; run_eval.py upgraded with --mode transition, per-category aggregation, MQ metric. Two-track eval (benchmark QA + internal transition). See Day 3 log. |
| 7 | Baseline 1 (Static Memory) smoke test | AI → Human debug | ⬜ | static_memory.py implemented; needs human to run smoke test |
| 8 | Baselines 2–4 | AI | ✅ | time_decay.py, heuristic_consolidation.py, llm_based.py implemented |
| 9 | Pilot MSTM checkpoint | Human (train) | ⬜ | model.py, train.py, inference.py implemented; label masking fixed; needs human to run training |
| 10 | Final MSTM training | Human (train) | ⬜ | |
| 11 | A-MEM / AgeMem reproduction | Human | ⬜ | |
| 12 | Master results table | AI → Human verify | ⬜ | |
| 13 | RQ1–RQ3 analysis | AI | ⬜ | |
| 14 | RQ4 analysis (key result) | AI → Human check | ⬜ | |
| 15 | Qualitative/error analysis | Human | ⬜ | |
| 16 | Draft full paper | AI → Human edit | 🔄 | Paper skeleton + 16 references written; sections 1-4, 6-7 drafted; results (sec 5) are placeholders |
| 17 | Review, format, submit | Human | ⬜ | |

---

## Known risks — live tracking

*Mirrors `CLAUDE.md` "Known risks" — update status as the sprint progresses.*

| Risk | Status | Notes |
|---|---|---|
| AgeMem/LightMem code unavailable | ⬜ | |
| Dataset quality issues from LLM generation | ⬜ | |
| SFT performance ceiling below RL | ⬜ | If confirmed, reframe paper toward cost-efficiency (RQ4) as headline |

---

## Blockers (active)

*List anything currently stuck. Remove once resolved (move a summary to the Log below).*

- (none yet)

---

## Log

*Append one entry per sync point or end-of-day. Newest at the top.*

### Day 3 — Eval Pipeline Upgrade + Doc Sync
- **Regenerated dataset verified:** Compression ratios fixed — abstraction 1.20→0.624, contradiction 0.817, consolidation 0.756, forgetting 0.486, update 0.882. All < 1.0. Sample quality confirmed.
- **Decision: Keep 2 benchmarks** (LoCoMo + LongMemEval). No 3rd benchmark — conference scope per CLAUDE.md.
- **Decision: Two-track evaluation.** (1) Benchmark QA = end-to-end utility, (2) Internal transition eval = direct per-operation evidence. Benchmark categories don't map to our operation types — the internal eval fills this gap.
- **No new code files.** Transition eval folded into `run_eval.py` via `--mode transition` flag. All new metrics in existing `metrics.py`.
- **metrics.py additions:** `rouge_l()`, `compute_per_category_metrics()`, `fact_preservation_rate()`, `transition_judge_score()`, `compute_transition_metrics()`, `compute_consistency_metrics_llm()`.
- **run_eval.py additions:** `--mode` flag (benchmark|transition), `--test_data` for transition mode, `evaluate_transition()` function, `_compute_memory_quality()` (AgeMem-style MQ), per-category aggregation in benchmark mode.
- **Docs synced:** TABLE.md (+Table A4), skeleton.md (§5 restructured), PROPOSAL.md (§8 two-track), TODOLIST.md (Phase 3 tasks), PROGRESS.md (this entry).
- **Prompt tightening:** consolidation, update, forgetting, contradiction prompts updated with compactness constraints and concise bullet-point examples. generate.py now supports `--api-key` and `--base-url` for third-party providers.
- **Human track:** Dataset regeneration with tightened prompts pending.

### Day 4 — A-MEM Pipeline Refactor
- **Eval pipeline refactored to A-MEM setup:** Conversation → Memory → chunk+embed → Retrieve Top-K → Foundation Model → Eval.
- **New file `src/eval/retriever.py`:** MemoryRetriever with all-MiniLM-L6-v2, chunking (bullet-split + 128-word guard), top-k cosine retrieval, tiktoken-based token length.
- **New metrics in `metrics.py`:** bleu_1, rouge_n, meteor, sbert_similarity, count_tokens_tiktoken, compute_full_qa_metrics (F1/BLEU-1/ROUGE-L/ROUGE-2/METEOR/SBERT), compute_average_ranking.
- **run_eval.py refactored:** Foundation model registry (GPT-4o-mini, GPT-4o, Qwen2.5-1.5B/3B, Llama-3.2-1B/3B), client factory for OpenAI + Ollama, retrieval pipeline, `--top-k`, `--k-sweep`, `--foundation-models` flags, `locomo_full` baseline (full dialogue, no memory system).
- **Loaders updated:** build_full_dialogue added to locomo.py + longmemeval.py (interleaves user+assistant turns).
- **MSTM fix:** inference.py temperature default changed to 0.0 (deterministic builds, avoids reproducibility bug across foundation-model sweep).
- **requirements.txt:** tiktoken added.
- **Baselines:** Reported numbers for A-MEM, LightMem, ReadAgent, MemoryBank, MemGPT. We run: MSTM + 4 existing + new locomo_full.
- **Scope override:** Foundation model sweep (6 models) is intentional — matches A-MEM setup, explicitly overrides CLAUDE.md "no multi-backbone sweep" guardrail.

### Day 2 — Dataset Fix + Paper Prep
- **Prompt fixes:** Rewrote abstraction and contradiction_resolution prompts to produce compact M′ (not longer than M+ΔM). Abstraction now keeps 1-2 representative episodes instead of all.
- **split_dataset.py:** Stratified 80/10/10 split with category balance, per-category output, dry-run mode, split summary.
- **MSTM training code review:** Fixed label masking — model now only trains on M′ generation, ignoring prompt tokens. Verified seq_length=2048 is sufficient (data max ~250 tokens).
- **Paper skeleton:** Full paper outline written (sections 1-7) with structured content for Introduction, Related Work, Method, Experimental Setup, and Discussion. Results section has placeholder tables.
- **References:** 16 references in BibTeX format across 6 categories: competing methods, memory systems, agents, benchmarks, SLMs/SFT, cognitive science.
- **Human track:** Regenerating abstraction.jsonl and contradiction_resolution.jsonl with fixed prompts.

### Day 1 — AI Track Implementation
- **Scoping decisions locked:** SLM = Qwen 2.5 0.5B, dataset = ~5,000 examples, generator = GPT-4o.
- **5 prompt templates written** (update, contradiction_resolution, consolidation, abstraction, forgetting) in `src/data_generation/prompts/`.
- **generate.py implemented** with GPT-4o API integration, category routing, validation, and batch generation.
- **Baselines 1-4 implemented with actual logic:**
  - static_memory.py: pass-through concatenation
  - time_decay.py: age-based pruning with regex date extraction
  - heuristic_consolidation.py: embedding-similarity merge (sentence-transformers with TF-IDF fallback)
  - llm_based.py: Mem0-style prompting pipeline with OpenAI-compatible API
- **Eval harness implemented:**
  - metrics.py: EM, F1, Recall@K, MRR, compression ratio, contradiction rate, redundancy rate, temporal accuracy, CostTracker
  - loaders/locomo.py: HuggingFace + local loading, QA extraction, memory building
  - loaders/longmemeval.py: HuggingFace + local loading, QA extraction, memory building
  - run_eval.py: full evaluation pipeline with method loading, benchmark routing, result saving
- **MSTM code implemented:**
  - model.py: Qwen2.5-0.5B wrapper with LoRA, prompt formatting, generation
  - train.py: SFT training loop with HuggingFace Trainer, config file support, cost tracking
  - inference.py: inference wrapper compatible with eval harness
- **Documentation updated:** requirements.txt, configs/mstm_sft.example.yaml, experiments/logs/README.md
- **Next:** Human track needs to (1) set up GPU environment, (2) run dataset generation, (3) run Baseline 1 smoke test, (4) train pilot MSTM checkpoint.

### Day 0 — Kickoff
- Project scaffolded: `PROPOSAL.md`, `TODOLIST.md`, `TABLE.md`, `WORKPLAN.md`, `CLAUDE.md`, repo structure created.
- Next: Day 1–2 setup (see `WORKPLAN.md`).