# CLAUDE.md

Orientation file for any AI assistant (or human) picking up this project. Read this first, then `PROPOSAL.md`, then `TODOLIST.md`.

## What this project is

**Learning Memory State Transition for Long-Term Conversational Agents.** We're training a small language model (SLM) to directly rewrite an agent's memory state — `T(M, ΔM) → M′` — as a single generative pass, instead of the more common approach of classifying discrete memory operations (add/update/delete) and applying them via heuristics or an RL-trained policy.

Full details: `PROPOSAL.md`. Task breakdown: `TODOLIST.md`. Paper table plan: `TABLE.md`.

## The two things that make this project different (don't lose these)

1. **Generative implicit transition, not discrete action selection.** Closest competing work (AgeMem, A-MEM) predicts *which* operation to apply. We generate the *result* directly — the operation is implicit in the output. This is the strongest argument specifically for Consolidation and Abstraction, where the correct output isn't a single-record edit.
2. **SFT, not RL.** AgeMem trains via a 3-stage RL curriculum (GRPO). We train via supervised fine-tuning on constructed (M, ΔM, M′) triplets — cheaper, more stable, easier to reproduce. We are explicit that this trades off a potentially lower performance ceiling for much lower training cost (this is RQ4, the paper's key result — see `TABLE.md` Table 2).

If any implementation decision seems to blur these two points, stop and reconsider — they're the entire novelty claim relative to `PROPOSAL.md` §2 (Related Work).

## Explicitly out of scope

- No Vietnamese-language dataset branch. This was considered and deliberately dropped — do not reintroduce it without the user's explicit request.
- No RL training for our own method. If someone suggests "let's also try RL for MSTM," that's a different paper — flag it rather than quietly implementing it.
- Not targeting a Q1 journal. Scoped as a conference-scale study (see `TABLE.md` header note). Don't add journal-scale scope (3+ benchmarks, mandatory multi-seed stats, multi-backbone sweeps) unless the user changes the target venue.
- **Exception (Day 4):** Foundation model sweep (6 models: GPT-4o-mini, GPT-4o, Qwen2.5-1.5B/3B, Llama-3.2-1B/3B) is explicitly enabled to match A-MEM's Evaluation Setup. This is a user-requested scope override, not scope creep.

## Execution order (see TODOLIST.md for full detail)

The build order is deliberate — **do not build MSTM before the easy baselines and eval pipeline exist**:

1. Scoping → lock proposal parameters (primary SLM, compute budget, venue)
2. Related work deep-read (AgeMem, LightMem, A-MEM — check code/checkpoint availability, this gates Phase 5)
3. Dataset construction (pilot small, review, then scale to 5,000–20,000 triplets)
4. **Eval pipeline + easy baselines first** (Static Memory as a smoke test, then Time-Decay/Heuristic/Mem0-style) — validates the harness before anything expensive is built on top of it
5. **Build MSTM** (SFT-train the SLM) — only after step 4's pipeline is trustworthy
6. **Hard baselines last** (A-MEM, AgeMem) — most reproduction-risk, saved for when a failure here won't block everything else
7. Analysis → Writing

## Repo structure

```
memory-state-transition/
├── CLAUDE.md              # this file
├── PROPOSAL.md            # research proposal — source of truth for RQs, method, contributions
├── TODOLIST.md            # phased task list with execution order and rationale
├── TABLE.md               # paper table plan (main body vs. appendix), placeholders marked TBD
├── data/
│   ├── raw/                # raw generated (M, ΔM, M′) triplets before review, LoCoMo/LongMemEval raw downloads
│   └── processed/          # cleaned, split (train/val/test) datasets ready for training/eval
├── src/
│   ├── data_generation/    # prompts + scripts to generate the triplet dataset
│   │   └── prompts/         # one prompt template per category (update, consolidation, abstraction, forgetting, contradiction)
│   ├── baselines/
│   │   ├── amem/            # A-MEM reproduction/adaptation
│   │   ├── agemem/          # AgeMem reproduction/adaptation (or reported-number fallback notes)
│   │   └── (static_memory.py, time_decay.py, heuristic_consolidation.py, llm_based.py go here directly)
│   ├── mstm/                # our method: SFT training, model wrapper, inference
│   └── eval/
│       ├── loaders/          # locomo.py, longmemeval.py — benchmark data loading
│       └── (metrics.py, run_eval.py go here directly)
├── experiments/
│   └── logs/                # running experiment log — config, results, date, per TODOLIST.md "Cross-cutting"
├── results/
│   ├── raw_outputs/         # raw model outputs / eval outputs per run
│   └── tables/              # generated tables filled in from results, matching TABLE.md structure
├── paper/
│   ├── main/                 # main paper draft sections
│   └── appendix/             # appendix sections (dataset stats, per-category breakdown, baseline implementation notes)
└── configs/                  # training/eval configs (yaml/json), one per experiment run
```

## Conventions

- **Dataset schema**: every triplet is `{"M": ..., "delta_M": ..., "M_prime": ..., "category": ...}` — category ∈ {update, contradiction_resolution, consolidation, abstraction, forgetting}. Keep this schema stable across `data/raw` and `data/processed`; downstream code assumes it.
- **Experiment logging**: every training or eval run gets an entry in `experiments/logs/` (config used, date, key results, git commit hash if applicable) before moving on to the next run. This is not optional — it's what prevents re-running duplicate experiments (a named risk in `TODOLIST.md`).
- **Baseline honesty**: for A-MEM and AgeMem specifically, always record in `src/baselines/*/README.md` (create if needed) whether a result is *reproduced* (ran ourselves, same hardware/data) or *reported* (taken from the original paper). This distinction must survive into `TABLE.md` Table A3 — do not blur it.
- **No placeholder numbers in `paper/`**: `TABLE.md` uses `TBD` intentionally as a staging area. Don't copy tables into `paper/` until real numbers exist.

## Known risks (carry these forward)

- AgeMem / LightMem code may not be public → Baseline 6 may end up as reported-numbers-only. Flag clearly, don't hide it.
- LLM-generated dataset may have quality issues → budget real time for manual review (Phase 2, Task 4), don't skip it under time pressure.
- SFT may have a real performance ceiling below RL → if this happens, the paper's framing should shift toward cost-efficiency (RQ4) as the headline, not "matches SOTA." This is an anticipated outcome, not a failure state.

## When in doubt

Check `PROPOSAL.md` §2 (Related Work) and §9 (Contributions) before making a scope decision — if a proposed change doesn't clearly support one of the three stated contributions, it's probably scope creep.
