# Tables for Paper

Placeholder tables matching `PROPOSAL.md`. Values marked `TBD` — fill in as experiments complete (`TODOLIST.md` Phase 3–6). Organized by where each table belongs: main paper body vs. appendix.

**Note on venue:** scoped for a conference-scale paper (per `PROPOSAL.md` §6), not a Q1 journal submission. A Q1 journal would typically require 3+ benchmarks, mandatory multi-seed significance testing, comparison across multiple SLM backbones, and likely a human evaluation table.

**Related Work — no table.** Feature-comparison matrices (system × mechanism) are a survey-paper convention, not a research-paper one. In the main text, Related Work should be prose with citations — state the positioning argument (generative rewrite vs. discrete action; SFT vs. RL) directly in 1–2 paragraphs rather than encoding it as a table row. If a compact reference table is still useful for the reader, it belongs in the appendix, not the main body (see Table A0 below, optional).

---

## Main Paper

### Table 1 — Main Results

*Results section. Both benchmarks combined into one table so the baseline list isn't repeated. Memory efficiency metrics included inline since they're read alongside QA performance. Per-category breakdown for benchmark-native categories (e.g., LoCoMo: single-hop, multi-hop, temporal, open-domain, adversarial) is computed by `compute_per_category_metrics` and reported in the detailed output.*

| Method | LoCoMo F1 | LoCoMo Recall@5 | LongMemEval F1 | LongMemEval Recall@5 | # Memory Records | Compression Ratio |
|---|---|---|---|---|---|---|
| Baseline 1 — Static Memory | TBD | TBD | TBD | TBD | TBD | 1.00 (ref.) |
| Baseline 2 — Time-Decay Forgetting | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline 3 — Heuristic Consolidation | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline 4 — LLM-Based (Mem0-style) | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline 5 — A-MEM | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline 6 — AgeMem | TBD | TBD | TBD | TBD | TBD | TBD |
| **MSTM (ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

*EM and MRR: report in text only if they diverge meaningfully from the F1/Recall@5 trend — otherwise redundant with this table.*

---

### Table 2 — Cost / Performance Trade-off (RQ4 — key result)

*Results/Discussion section. The paper's central differentiating claim — must be in the main body, not deferred to an appendix.*

| Method | Training Method | GPU-Hours | # Training Examples | Avg. F1 (both benchmarks) | Inference Latency / Update |
|---|---|---|---|---|---|
| Baseline 5 — A-MEM | Heuristic (no training) | 0 | 0 | TBD | TBD |
| Baseline 6 — AgeMem | RL (3-stage GRPO) | TBD | TBD | TBD | TBD |
| **MSTM (ours)** | **SFT** | **TBD** | **TBD** | **TBD** | **TBD** |

*If AgeMem is not reproduced under identical hardware, mark GPU-hours as "reported" vs. "measured" in a footnote.*

---

### Table 3 — Temporal Reasoning & Consistency Breakdown

*Results section, RQ3. Only methods with LLM-driven or learned update logic are informative — Baselines 2–3 are purely heuristic and omitted.*

| Method | Temporal QA Acc. | Update-Sensitive Acc. | Contradiction Rate ↓ | Redundancy Rate ↓ |
|---|---|---|---|---|
| Baseline 1 — Static Memory | TBD | TBD | TBD | TBD |
| Baseline 4 — LLM-Based (Mem0-style) | TBD | TBD | TBD | TBD |
| Baseline 5 — A-MEM | TBD | TBD | TBD | TBD |
| Baseline 6 — AgeMem | TBD | TBD | TBD | TBD |
| **MSTM (ours)** | **TBD** | **TBD** | **TBD** | **TBD** |

*If space is tight, this can move to the appendix and be summarized in one sentence in the main text — keep here only if RQ3 results are a notable finding (e.g., a clear win or a surprising failure).*

---

## Appendix

### Table A0 — Related Work Comparison *(optional)*

*Only include if a reviewer/advisor specifically wants a quick-reference table alongside the prose Related Work section. Not required.*

| System | Update Mechanism | Consolidation/Abstraction | Training Method |
|---|---|---|---|
| MemGPT / Letta | Manual paging | No | None (heuristic) |
| MemoryBank | Rule-based update + decay | No | None (heuristic) |
| Mem0 / Mem0g | LLM-prompted extraction | Partial (graph merge) | None (prompting) |
| A-MEM | Supersede detection (discrete) | Linking only | None (heuristic) |
| Zep | Temporal graph edit | No | None (heuristic) |
| LightMem | SLM-routed, LLM-consolidated | Offline, LLM-based | SLM: routing only |
| AgeMem | Discrete action (RL policy) | Via discrete action | RL (GRPO) |
| **MSTM (ours)** | **Generative rewrite** | **Generative rewrite** | **SFT** |

---

### Table A1 — Dataset Statistics

*Appendix or Data Statement. Main text needs only a one-sentence summary (e.g., "N examples across 5 categories").*

| Category | # Examples | % of dataset | Avg. len(M) | Avg. len(ΔM) | Avg. len(M′) |
|---|---|---|---|---|---|
| Update | TBD | TBD | TBD | TBD | TBD |
| Contradiction Resolution | TBD | TBD | TBD | TBD | TBD |
| Consolidation | TBD | TBD | TBD | TBD | TBD |
| Abstraction | TBD | TBD | TBD | TBD | TBD |
| Forgetting | TBD | TBD | TBD | TBD | TBD |
| **Total** | **TBD** | **100%** | — | — | — |

**Splits:** Train `TBD` / Val `TBD` / Test `TBD` (split by source conversation, no leakage)

---

### Table A2 — Per-Category Performance (Benchmark-Native Categories)

*Appendix. QA performance broken down by the benchmark's own question categories — not our training operation categories. This shows how methods perform on different types of memory questions. The internal transition-quality eval (Table A4) covers per-operation performance.*

| Category | Baseline 5 (A-MEM) F1 | Baseline 6 (AgeMem) F1 | MSTM (ours) F1 |
|---|---|---|---|
| Single-hop | TBD | TBD | TBD |
| Multi-hop | TBD | TBD | TBD |
| Temporal | TBD | TBD | TBD |
| Open-domain | TBD | TBD | TBD |
| Adversarial | TBD | TBD | TBD |

---

### Table A3 — Baseline Implementation Summary

*Appendix, Reproducibility section. Required disclosure since Baselines 5–6 may not be identical-condition reproductions — but this level of detail never belongs in the main body.*

| Baseline | Implementation Source | Reproduced or Reported? | Notes |
|---|---|---|---|
| 1 — Static Memory | Implemented in-house | Reproduced | Pass-through, trivial |
| 2 — Time-Decay Forgetting | Implemented in-house | Reproduced | Age threshold: TBD |
| 3 — Heuristic Consolidation | Implemented in-house | Reproduced | Similarity threshold: TBD |
| 4 — LLM-Based (Mem0-style) | Implemented in-house / Mem0 codebase | TBD | Backbone LLM used: TBD |
| 5 — A-MEM | Public code (if available) | TBD | — |
| 6 — AgeMem | Public code (if available) / reported numbers | TBD | — |

---

### Table A4 — Internal Transition-Quality Eval (Per Operation)

*Appendix. Direct evidence for the headline claim — measures how well each method reproduces gold M′ on the held-out test split, grouped by our 5 training operation categories. Metrics: ROUGE-L (R-L), token F1, fact preservation rate (FP), transition judge (TJ), compression ratio (CR). The predicted compression ratio is compared against the gold compression ratio to show fidelity to the expected compact/expand behavior.*

**All methods (full comparison):**

| Operation | Metric | B1 (Static) | B2 (TimeDecay) | B3 (Heuristic) | B4 (LLM) | B5 (A-MEM) | **MSTM (Ours)** |
|---|---|---|---|---|---|---|---|
| Update | R-L / F1 / FP / TJ / CR | TBD | TBD | TBD | TBD | TBD | **TBD** |
| Contradiction | R-L / F1 / FP / TJ / CR | TBD | TBD | TBD | TBD | TBD | **TBD** |
| Consolidation | R-L / F1 / FP / TJ / CR | TBD | TBD | TBD | TBD | TBD | **TBD** |
| Abstraction | R-L / F1 / FP / TJ / CR | TBD | TBD | TBD | TBD | TBD | **TBD** |
| Forgetting | R-L / F1 / FP / TJ / CR | TBD | TBD | TBD | TBD | TBD | **TBD** |

**Simplified view (for main text if space is tight):**

| Operation | MSTM Transition Judge | Best Baseline TJ | Δ |
|---|---|---|---|
| Consolidation | TBD | TBD | TBD |
| Abstraction | TBD | TBD | TBD |
| Forgetting | TBD | TBD | TBD |
| Update | TBD | TBD | TBD |
| Contradiction | TBD | TBD | TBD |

*Expected pattern: largest Δ on Consolidation and Abstraction rows — this is the direct evidence for the paper's key claim. If this pattern doesn't hold, the paper's main argument weakens significantly.*

---

## Cut entirely (and why)

- ~~Separate LoCoMo and LongMemEval tables~~ → merged into Table 1
- ~~Separate Memory Efficiency table~~ → merged into Table 1
- ~~Dataset size ablation table~~ → only add if actually run
- ~~SLM backbone comparison table~~ → proposal specifies one primary SLM; only add if multiple are trained
