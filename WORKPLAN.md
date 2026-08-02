# Work Plan — 1–2 Week Human + AI Sprint

Two parallel tracks. **AI** = you using Claude/an AI assistant for generation, boilerplate, and pipeline code. **Human** = you, for judgment calls, debugging, and decisions. Sync points are where one track's output blocks the other — don't skip them.

**Reality check first:** the full scope in `PROPOSAL.md`/`TODOLIST.md` (6 baselines incl. AgeMem's RL reproduction, 5,000–20,000 examples, 2 benchmarks) is tight for 1–2 weeks. This plan assumes the fallback from `TODOLIST.md` Task 1 kicks in around Day 5–6 if needed: **cut AgeMem reproduction → reported-numbers-only**, and **cap dataset at ~5,000 examples** rather than 20,000. Both cuts are flagged below at the point they'd need to happen.

---

## Days 1–2 — Setup & Kickoff (parallel from hour one)

| Track | Work |
|---|---|
| **AI** | Deep-read AgeMem, LightMem, A-MEM papers and summarize mechanisms; draft Related Work prose; write the 5 category-specific dataset generation prompts (`src/data_generation/prompts/`); scaffold all baseline stub code with actual logic for Baselines 1–4 (these are simple enough to fully draft from spec) |
| **Human** | Set up GPU environment; clone A-MEM and AgeMem repos, read their code, confirm what's actually reproducible; pick and download LoCoMo + LongMemEval; lock the primary SLM (Qwen 0.5B recommended for fastest iteration) |
| **Sync (end of Day 2)** | Human confirms AI's dataset prompts look right and AgeMem/A-MEM reproducibility verdict (reproducible / reported-only) — this decides Phase 5 scope now instead of later |

---

## Days 3–4 — Dataset + Eval Harness (parallel)

| Track | Work |
|---|---|
| **AI** | Generate pilot dataset batch (~500 examples) using approved prompts; write eval harness code — data loaders for both benchmarks, all metric functions (`src/eval/`) |
| **Human** | Manually review the pilot batch for quality/noise/category balance (this genuinely needs a human eye — AI grading its own generations is unreliable); in parallel, get A-MEM baseline running end-to-end on a small sample |
| **Sync (end of Day 4)** | Human approves dataset prompts (revise with AI if pilot has issues) → AI scales to full dataset generation overnight/in background |

---

## Days 5–6 — Scale Up + Pipeline Validation

| Track | Work |
|---|---|
| **AI** | Generate full dataset (target size decided at Day 2 sync — 5,000 if compressed); implement Baseline 1 (Static Memory) fully and run it through the eval harness as the smoke test; implement Baselines 2–4 and run them |
| **Human** | Spot-check the full generated dataset (10% sample); fix any eval harness bugs the smoke test surfaces (this is debugging — needs human judgment on *why* something's wrong, not just AI re-generation); finalize A-MEM baseline results |
| **Sync (end of Day 6)** | **Scope decision checkpoint.** If AgeMem reproduction isn't working by now, cut it to reported-numbers-only and move on — don't let it block Days 7+ |

---

## Days 7–9 — Train MSTM (the critical path)

| Track | Work |
|---|---|
| **AI** | Write SFT training script (`src/mstm/train.py`) and inference wrapper; draft hyperparameter sweep config options |
| **Human** | Run training (needs human-in-the-loop for GPU babysitting, catching divergence, deciding when a run is worth continuing); make the actual hyperparameter decisions from AI's suggested ranges; log every run in `experiments/logs/` |
| **Sync (Day 7)** | Pilot checkpoint (trained on pilot dataset) run through eval pipeline first — confirm it beats Baseline 1 before committing GPU time to a full training run |
| **Sync (Day 9)** | Final MSTM checkpoint selected; run through full eval harness |

This is the track most likely to eat extra days if training is unstable — treat Days 7–9 as the buffer-sensitive block.

---

## Days 10–11 — Remaining Baselines + Results Assembly

| Track | Work |
|---|---|
| **AI** | If AgeMem is still in scope: draft integration/adaptation code. If not: pull AgeMem's reported numbers from the paper into the results table with clear "reported, not reproduced" flag; populate `TABLE.md` Table 1–3 with all collected results |
| **Human** | Finish/debug AgeMem if attempting it (highest-risk task — this is why it's scheduled last, per `TODOLIST.md` Phase 5 rationale); verify all numbers in the results table against raw output logs |
| **Sync (end of Day 11)** | Master results table locked. No more experiment changes after this point unless something is clearly broken |

---

## Days 12–13 — Analysis & Writing

| Track | Work |
|---|---|
| **AI** | Draft RQ1–RQ3 analysis from the results table; draft RQ4 cost/performance analysis and plot; draft full paper sections (Intro, Method, Experimental Setup, Results) from `PROPOSAL.md` + final numbers |
| **Human** | Write/edit Discussion & Limitations (the SFT-vs-RL trade-off framing needs a human voice, since it's the paper's central honest caveat); do qualitative error analysis by reading actual failure cases; sanity-check that AI's drafted claims match the actual numbers (AI will confidently overstate results if not checked) |
| **Sync (end of Day 13)** | Full draft assembled |

---

## Day 14 — Review & Submit (buffer day if 2-week timeline)

| Track | Work |
|---|---|
| **AI** | Proofread pass, citation formatting check, consistency check between text and tables |
| **Human** | Final read-through and judgment call on any remaining rough edges; submit |

---

## Division-of-labor principle (why it's split this way)

- **AI gets:** anything with a clear spec and no need for real-world judgment — code from a docstring, prompts from a schema, prose from a locked results table, literature summarization.
- **Human gets:** anything where being wrong is expensive and hard to detect automatically — dataset quality (AI can't reliably self-grade its own generations), debugging other people's research code (A-MEM/AgeMem), hyperparameter/training judgment calls, and the paper's core honest framing (SFT-vs-RL trade-off, limitations).
- **1-week version:** compress by cutting Days 12–14 into 2 days and accepting a rougher draft, or by cutting AgeMem reproduction at the Day 2 sync instead of Day 6 (saves ~2 days of Human track time that would otherwise go to debugging it).
