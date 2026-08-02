# TODO List — Memory State Transition Project

Ordered task list, grouped into phases. Each numbered task is a unit of work with a short description and the concrete steps it involves.

---

## Phase 0 — Scoping

### 1. Lock the proposal and project parameters
Finalize `PROPOSAL.md` as the reference document, and pin down the parameters everything else depends on.
- Confirm target venue (affects page limit, rigor expected)
- Pick primary SLM: Qwen 0.5B / Phi-3 Mini / SmolLM (others only as ablation if time allows)
- Confirm GPU budget and hard deadline
- Write a fallback plan: what gets cut first if time runs short (e.g., drop Forgetting category before Update/Consolidation; drop AgeMem reproduction before A-MEM)

---

## Phase 1 — Related Work

### 2. Deep-read the closest competing papers
Read AgeMem, LightMem, and A-MEM in full — not just abstracts — since Phase 5 depends on what you learn here.
- For each: extract exact mechanism (action space / detection rule / online-offline split), training procedure, and reported benchmark numbers
- Check whether code or checkpoints are publicly available for each — this determines if Baselines 5–6 will be true reproductions or reported-number comparisons
- Skim MemoryBank, Mem0, Zep for reusable evaluation protocol details (metrics, splits)

### 3. Draft the Related Work section
- Reuse and tighten `PROPOSAL.md` Section 2 into paper-ready prose
- Make sure the positioning argument (generative rewrite vs. discrete action; SFT vs. RL) is stated explicitly, not just implied by the comparison table

---

## Phase 2 — Dataset Construction

### 4. Design and pilot the triplet dataset
Don't scale up until this pilot passes review — bad prompts compound expensively at full scale.
- Define the (M, ΔM, M′) JSON schema and max lengths
- Write generation prompts for each category: Update, Contradiction Resolution, Consolidation, Abstraction, Forgetting
- Generate a ~500-example pilot batch
- Manually review for quality, label noise, and category balance; revise prompts until clean

### 5. Scale up and finalize the dataset
- Generate the full target of 5,000–20,000 examples with the validated prompts
- Manual spot-check a defined sample (e.g., 10%)
- Split into train/val/test with no leakage (split by source conversation, not individual triplet)
- Document final stats: category distribution, avg length of M/ΔM/M′

---

## Phase 3 — Evaluation Pipeline & Easy Baselines

*Build a trustworthy eval pipeline before spending time on MSTM training or the hard baselines.*

### 6. Build the evaluation harness
Two-track evaluation: (1) benchmark QA eval on LoCoMo + LongMemEval, (2) internal transition-quality eval on our test split.
- Set up data loading + QA evaluation for LoCoMo and LongMemEval
- Implement all metrics:
  - **QA**: LLM-Judge Accuracy (primary, GPT-4o), Token-Level F1, EM, ROUGE-L
  - **Memory Quality (MQ)**: AgeMem-style LLM evaluation of M′ vs ground truth
  - **Retrieval**: Recall@K, MRR
  - **Memory efficiency**: record count, token count, compression ratio
  - **Consistency**: contradiction rate, redundancy rate (heuristic + LLM-based)
  - **Temporal**: temporal QA accuracy, update-sensitive accuracy
  - **Per-category**: benchmark-native category breakdown (LoCoMo: single-hop/multi-hop/temporal/open-domain/adversarial; LongMemEval: IE/MR/KU/TR/ABS)
  - **Transition-quality** (internal test-split eval, `--mode transition`): ROUGE-L, fact preservation rate, transition judge score, compression fidelity — per operation category
- Hook up training-cost tracking (GPU-hours, examples seen, inference latency) now, even though it's populated later
- No new code files — transition eval folded into `run_eval.py` via `--mode` flag

### 7. Implement Baseline 1 (Static Memory) as a pipeline smoke test
- Implement as a pure pass-through (no modification)
- Run it through the full harness end-to-end
- Fix any bugs in loaders/metrics/scoring this surfaces — don't proceed until this baseline's numbers look sane

### 8. Implement and run Baselines 2–4
None require training; run all three through the now-verified pipeline for early reference numbers.
- Baseline 2 — Time-Decay Forgetting: age-based pruning rule
- Baseline 3 — Heuristic Consolidation: embedding-similarity merge threshold
- Baseline 4 — LLM-Based Memory Management: Mem0-style prompting pipeline

---

## Phase 4 — Build Our Method (MSTM)

*Train MSTM against the trusted baseline numbers from Phase 3, on a pipeline already known to be correct.*

### 9. Train and sanity-check a pilot MSTM checkpoint
- Set up SFT training pipeline for the chosen SLM
- Define the input/output prompt format for T(M, ΔM) → M′
- Train an initial checkpoint on the pilot dataset only
- Run it through the eval pipeline — confirm it beats Baseline 1 and lands in a sane range vs. Baselines 2–4 before committing to a full run

### 10. Train the final MSTM and tune hyperparameters
- Train on the full dataset
- Sweep learning rate, epochs, LoRA vs. full fine-tune
- Log training cost metrics (GPU-hours, examples seen) needed for RQ4
- Run the final model through the eval pipeline and record results

---

## Phase 5 — Hard / Risky Baselines (A-MEM, AgeMem)

*Save the most expensive, reproduction-risky baselines for last — pipeline is proven and MSTM already has results, so a failure here doesn't block the rest of the project.*

### 11. Reproduce or adapt A-MEM and AgeMem
- Check public code/checkpoint availability first (from Task 2 notes)
- Reproduce and run through the eval pipeline where feasible
- If infeasible (most likely for AgeMem, given its RL training complexity) fall back to reported benchmark numbers, with a clearly flagged caveat in the paper

### 12. Assemble the master results table
- Collect all six baselines plus MSTM across both benchmarks into one table
- Add significance tests / error bars if budget allows multiple training seeds

---

## Phase 6 — Analysis

### 13. Answer RQ1–RQ3 (general effectiveness)
- RQ1: static vs. learned transition, QA performance comparison
- RQ2: memory size/redundancy reduction vs. baselines
- RQ3: temporal reasoning accuracy, broken down by question type

### 14. Answer RQ4 (cost/performance trade-off) — the paper's key result
Give this more depth than RQ1–3; it's the main differentiating claim.
- Plot performance vs. training cost: MSTM (SFT) vs. AgeMem (RL) vs. A-MEM (discrete detection)
- Use internal transition eval (per-operation) to show where generative rewrite is strongest — the consolidation/abstraction rows should show the largest Δ over baselines
- Benchmark QA per-category breakdown provides supporting evidence from downstream utility perspective

### 15. Qualitative and error analysis
- Pull sample (M, ΔM, M′) transitions, including failure cases
- Identify where generative rewrite fails relative to discrete-action baselines
- If time allows, run a dataset-size ablation to see how performance scales with training data

---

## Phase 7 — Writing

### 16. Draft the full paper
Introduction → Related Work (Task 3) → Method → Experimental Setup → Results → Discussion/Limitations → Conclusion.
- In Discussion/Limitations, be explicit about the SFT performance ceiling vs. RL — frame it as a known, expected trade-off, not a weakness to hide

### 17. Review, format, and submit
- Internal review pass (self or advisor/co-author)
- Check page limit and formatting for target venue
- Finalize citations, proofread
- Submit

---

## Cross-cutting (ongoing throughout)

- Keep a running experiment log (config, results, date) to avoid re-running duplicate experiments
- Version-control the dataset and code (git repo) from Phase 2 onward
- Watch three known risks: AgeMem/LightMem code unavailable (→ fallback to reported numbers), dataset quality issues from LLM generation (→ budget extra manual-review time), SFT ceiling too low vs. RL (→ be ready to reframe the paper's angle as cost-efficiency rather than "matches SOTA")
