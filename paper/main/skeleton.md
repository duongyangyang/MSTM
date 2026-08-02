# Paper Skeleton — Memory State Transition for Long-Term Conversational Agents

**Target venue:** Conference (ACL / EMNLP / NeurIPS scale)
**Page budget:** 8 pages + appendix
**References:** 16 (target: 12–18)
**Status:** Draft skeleton — results sections are placeholders until experiments complete

---

## 1. Introduction

### Hook (1 paragraph)
Long-term conversational agents — from personal AI assistants to customer support chatbots — must maintain coherent memory of user interactions across days, weeks, and months. However, as conversation history grows, fixed-context LLMs face a fundamental tension: retaining all history exceeds context window limits, while naive compression or retrieval risks losing critical information.

### Problem Statement (1 paragraph)
Recent work addresses this through **memory management modules** that decide which memory operations to apply (add, update, delete, retrieve). These methods fall into three paradigms:
- **Discrete action selection** (AgeMem): the agent learns to pick from a fixed set of operations via RL
- **Heuristic detection** (A-MEM, MemoryBank): rule-based or LLM-prompted detection of when to link, supersede, or consolidate
- **System-level routing** (LightMem, MemGPT): multi-component architectures that route information between memory tiers

All three paradigms share a common assumption: memory management is an **operation classification problem** — predict *which action*, then apply it deterministically.

### Our Claim (1 paragraph)
We argue that for key memory operations — particularly **consolidation** and **abstraction** — the right output is not a discrete action label but a **directly rewritten memory state**. When a user's dozen scattered remarks about cooking reveal a pattern ("this person is a serious Italian food enthusiast"), no single add/update/delete operation captures this. The transition is **generative and implicit**.

### Our Method (1 paragraph)
We propose **MSTM (Memory State Transition Model)**: a small language model (0.5B parameters) trained via supervised fine-tuning to directly generate the evolved memory state — T(M, ΔM) → M′. This is a single generative pass, not a sequence of discrete operations. Our training uses constructed (M, ΔM, M′) triplets across five memory operation categories.

### Contributions (bullet list)
1. **Generative implicit transition**: We formalize memory management as a text-to-text generation problem, where the model directly outputs the new memory state rather than classifying operations.
2. **SFT over RL**: We show that supervised fine-tuning on a 0.5B model achieves competitive results with RL-trained 7B models at a fraction of the training cost.
3. **Comprehensive evaluation**: We benchmark against 6 baselines on LoCoMo and LongMemEval, with detailed per-category analysis showing where generative rewrite excels (consolidation, abstraction) and where it falls short.

---

## 2. Related Work

### 2.1 Memory Systems for LLM Agents

Early work on LLM agent memory focused on retrieval-augmented architectures. MemGPT [memgpt2023] introduced OS-inspired virtual memory management, paging conversation history between context window (main memory) and external storage. Generative Agents [generativeagents2023] proposed a memory stream with reflection-based consolidation, where agents periodically synthesize higher-level observations from raw memory records. MemoryBank [memorybank2024] incorporated Ebbinghaus-inspired forgetting curves [ebbinghaus1885] to manage memory retention based on recency and importance.

More recent systems have adopted structured memory representations. Zep [zep2025] uses a temporal knowledge graph with bitemporal edge annotation to track both event time and ingestion time. Mem0 [mem02025] provides a production-oriented memory layer with semantic consolidation and intelligent forgetting. These systems share a common pattern: memory operations are either rule-based or LLM-prompted, not learned from data.

### 2.2 Learned Memory Management — Discrete Action Paradigm

The closest prior work to ours learns memory management from data, but through discrete action selection.

**AgeMem** [agemem2026] trains an LLM agent to select from a fixed set of 6 memory tools (ADD, UPDATE, DELETE, RETRIEVE, SUMMARY, FILTER) using a 3-stage progressive RL curriculum with GRPO. The backbone is a 7B model trained on Trinity-RFT. AgeMem demonstrates that learned memory management outperforms heuristic baselines, but at substantial training cost (RL on 7B parameters) and with the limitation that the action space is discrete — the model cannot express nuanced transitions that don't map cleanly to a single operation.

**A-MEM** [amem2025] takes a Zettelkasten-inspired approach: each new memory triggers note construction, link generation, and supersede detection via LLM prompting (GPT-4o-mini). No training is required, but memory evolution is limited to discrete detection decisions (link/no-link, supersede/no-supersede).

**LightMem** [lightmem2026] uses 3 specialized SLMs for online memory operations (controller, selector, writer) and a large-context LLM for offline consolidation. This is a system architecture contribution — the SLMs perform routing and filtering, not learned memory transitions.

### 2.3 Our Positioning

| Dimension | AgeMem | A-MEM | LightMem | **MSTM (Ours)** |
|---|---|---|---|---|
| Memory operation | Discrete action (6 tools) | Discrete detection + linking | SLM-routed, LLM-consolidated | **Generative rewrite** |
| Training method | RL (3-stage GRPO) | None (heuristic + LLM) | None (system design) | **SFT** |
| Model size | 7B/4B | GPT-4o-mini (API) | Multiple SLMs + LLM | **0.5B (local)** |
| Training cost | High (RL on 7B) | Zero (API calls) | Zero (API calls) | **Low (SFT on 0.5B)** |

Our work is the first to formulate memory state transition as a generative problem and to train an SLM for this task via SFT. The key difference from all prior work: we generate the *result* of memory operations, not the operations themselves.

### 2.4 Background: SFT and SLMs

Our method builds on two established lines of work. **Supervised fine-tuning (SFT)** [instructgpt2022] has proven effective for adapting LLMs to follow instructions. **Small language models (SLMs)** have rapidly closed the performance gap with larger models [qwen25techreport2024], making them viable for task-specific fine-tuning. We use LoRA [lora2022] for parameter-efficient adaptation of the 0.5B backbone.

### 2.5 Cognitive Science Foundations

The distinction between short-term and long-term memory [atkinsonshiffrin1968] and the dynamics of forgetting [ebbinghaus1885] provide the theoretical foundation for our work. Our five memory operation categories (update, contradiction resolution, consolidation, abstraction, forgetting) are grounded in established cognitive processes.

---

## 3. Method: Memory State Transition Model (MSTM)

### 3.1 Problem Formulation

Let M be the current memory state (a structured text record of what the agent knows about the user), and ΔM be new information arriving from the latest interaction. The memory state transition function T maps:

```
T(M, ΔM) → M′
```

where M′ is the evolved memory state after incorporating ΔM. Unlike prior work that decomposes T into discrete sub-operations, we model T directly as a generative process.

### 3.2 Model Architecture

We fine-tune a Qwen 2.5 0.5B model [qwen25techreport2024] as the backbone, with LoRA adapters [lora2022] (r=16, α=32) applied to all linear projection layers. The model receives a formatted prompt:

```
<|im_start|>system
You are a memory state transition model. Given the current memory state (M)
and new information (ΔM), produce the updated memory state (M′).
Update, consolidate, abstract, or forget as needed.
Output only the new memory state.<|im_end|>
<|im_start|>user
CURRENT MEMORY (M):
{M}

NEW INFORMATION (ΔM):
{delta_M}

Produce the UPDATED MEMORY STATE (M′):<|im_end|>
<|im_start|>assistant
```

The model generates M′ autoregressively. No operation classification head, no action space — the operation is implicit in the generated output.

### 3.3 Training Data

We construct a synthetic dataset of (M, ΔM, M′) triplets using GPT-4o as the generator. Each triplet belongs to one of five categories:

| Category | Definition | Example |
|---|---|---|
| **Update** | New info replaces/adds to existing facts | User moves from NYC → Chicago |
| **Contradiction Resolution** | New info conflicts with existing; resolve with temporal/rationale annotation | "I'm lactose intolerant" → "Actually I outgrew it" |
| **Consolidation** | Multiple related facts merge into a single coherent record | 5 cooking experiences → "Skilled Italian cook" |
| **Abstraction** | Episodic facts → higher-level pattern/trait | Scattered behaviors → "Strong mentor identity" |
| **Forgetting** | Remove obsolete/trivial/superseded information | Old trip plans, completed tasks, redundant facts |

Dataset size: ~5,000 examples (1,000 per category). Split: 80/10/10 train/val/test, stratified by category.

**Key design principle:** For consolidation and abstraction, M′ is **more compact** than M — the generative transition compresses information, not just reorganizes it. For forgetting, M′ removes records entirely. Only for update and contradiction resolution does M′ maintain similar length to M.

### 3.4 Training Procedure

We train via standard causal language modeling with label masking: only the M′ portion of the sequence contributes to the loss. Training uses the HuggingFace Trainer with:
- Learning rate: 2e-5, cosine schedule with 10% warmup
- Batch size: 4 × 4 gradient accumulation = effective 16
- 3 epochs, FP16 mixed precision
- Max sequence length: 2048 tokens (data max ~250 tokens)

Training cost is tracked via GPU-hours and examples processed — these feed into RQ4.

### 3.5 Inference

At inference time, the model receives the same prompt format without M′. Generation uses greedy decoding (temperature=0) for deterministic evaluation, with max_new_tokens=512.

---

## 4. Experimental Setup

### 4.1 Benchmarks

We evaluate on two conversational memory benchmarks:

- **LoCoMo** [locomo2024]: Long-term conversational memory with multi-session dialogues (avg 300 turns, 9K tokens across up to 35 sessions). Questions span single-hop, multi-hop, temporal, open-domain, and adversarial categories.
- **LongMemEval** [longmemeval2024]: Interactive memory benchmark with 500 questions across 6 categories, testing memory over 50+ sessions (LongMemEval-S: ~115K tokens).

### 4.2 Baselines

| # | Baseline | Type | Description |
|---|---|---|---|
| 1 | Static Memory | Heuristic | Pass-through: M′ = M + ΔM (no modification) |
| 2 | Time-Decay Forgetting | Heuristic | Age-based pruning using regex date extraction |
| 3 | Heuristic Consolidation | Heuristic | Embedding-similarity merge (sentence-transformers) |
| 4 | LLM-Based (Mem0-style) | Prompted | GPT-4o prompted to manage memory with system instructions |
| 5 | A-MEM [amem2025] | Reproduced | Zettelkasten-inspired memory linking + supersede detection |
| 6 | AgeMem [agemem2026] | Reported† | 3-stage RL with GRPO, discrete action space |

† AgeMem results are reported from the original paper, not reproduced. Benchmarks differ (agent-task vs. conversational memory), and RL training on 7B models is infeasible within our compute budget.

### 4.3 Metrics

- **QA Accuracy**: LLM-judge scoring (GPT-4o reader) — correct/partial/incorrect
- **Exact Match (EM)** and **Token-level F1**: standard generation metrics
- **Recall@K** and **MRR**: retrieval quality
- **Compression Ratio**: |M′| / (|M| + |ΔM|)
- **Contradiction Rate**: frequency of conflicting records in M′
- **Redundancy Rate**: duplicate or near-duplicate information in M′
- **Temporal Accuracy**: correct temporal ordering in multi-session recall
- **Training Cost**: GPU-hours, examples processed (for RQ4)

### 4.4 Implementation Details

- MSTM backbone: Qwen 2.5 0.5B with LoRA (r=16, α=32)
- Dataset generation: GPT-4o (temperature=0.8)
- Evaluation reader: GPT-4o (temperature=0)
- Hardware: [TBD — fill in based on actual training]

---

## 5. Results

> **⚠️ PLACEHOLDER — All numbers below are TBD. Fill in after experiments complete.**

### 5.1 Benchmark QA Results (End-to-End Utility)

**Table 1 — Main Results.** QA performance on LoCoMo and LongMemEval with memory efficiency metrics. Per-category benchmark-native breakdown (LoCoMo: single-hop, multi-hop, temporal, open-domain, adversarial; LongMemEval: IE, MR, KU, TR, ABS) reported in the detailed output and Appendix A2.

| Method | LoCoMo F1 | LoCoMo LLM-Judge | LoCoMo MQ | LongMemEval F1 | LongMemEval LLM-Judge | LongMemEval MQ | CompR |
|---|---|---|---|---|---|---|---|
| Static Memory (B1) | TBD | TBD | TBD | TBD | TBD | TBD | 1.00 |
| Time-Decay (B2) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Heuristic Consolidation (B3) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LLM-Based (B4) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A-MEM (B5) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AgeMem (B6)† | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **MSTM (Ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

† Reported numbers from original paper; benchmarks differ.

### 5.2 Internal Transition-Quality Results (Per-Operation — Direct Evidence)

This section evaluates how well each method reproduces gold M′ on the held-out test split, grouped by our 5 training operation categories. This is the direct evidence for the paper's central claim that generative rewrite excels at consolidation and abstraction.

**Table 2 — Per-Operation Transition Quality (simplified view).** Core metrics: ROUGE-L (R-L), token F1, fact preservation rate (FP), transition judge (TJ), compression ratio (CR). Full per-baseline breakdown in Table A4.

| Operation | MSTM R-L | MSTM TJ | Best Baseline TJ | Δ TJ (MSTM − Best) |
|---|---|---|---|---|
| Update | TBD | TBD | TBD | TBD |
| Contradiction Resolution | TBD | TBD | TBD | TBD |
| Consolidation | TBD | TBD | TBD | **TBD** |
| Abstraction | TBD | TBD | TBD | **TBD** |
| Forgetting | TBD | TBD | TBD | TBD |

**Expected pattern:** MSTM should show the largest Δ on Consolidation and Abstraction rows — these are the operations where generative rewrite is most advantageous over discrete action selection or heuristic merge. If this pattern does not hold, the paper's main argument weakens significantly.

### 5.3 Cost/Performance Trade-off (RQ4 — Key Result)

| Method | GPU-Hours | Examples | Model Size | LoCoMo F1 | Performance/Cost |
|---|---|---|---|---|---|
| AgeMem (RL) | TBD | TBD | 7B | TBD | TBD |
| A-MEM (API) | 0 | N/A | GPT-4o-mini | TBD | TBD |
| LLM-Based (API) | 0 | N/A | GPT-4o | TBD | TBD |
| **MSTM (SFT)** | **TBD** | **TBD** | **0.5B** | **TBD** | **TBD** |

**Expected pattern:** MSTM should achieve competitive performance at significantly lower training cost than AgeMem (RL on 7B).

### 5.4 Memory Efficiency Metrics (Table 4)

| Method | Compression Ratio | Redundancy Rate | Contradiction Rate | Temporal Accuracy |
|---|---|---|---|---|
| Static Memory | TBD | TBD | TBD | TBD |
| Time-Decay | TBD | TBD | TBD | TBD |
| MSTM | TBD | TBD | TBD | TBD |

---

## 6. Discussion

### 6.1 RQ1: Static vs. Learned Transition
[TBD — Does MSTM outperform the static/heuristic baselines?]

### 6.2 RQ2: Memory Size and Redundancy
[TBD — Does generative rewrite produce more compact, less redundant memory states?]

### 6.3 RQ3: Temporal Reasoning
[TBD — How does MSTM handle temporal ordering compared to baselines?]

### 6.4 RQ4: Cost/Performance Trade-off (Key Result)
[TBD — This is the paper's headline finding. We expect to show that SFT on a 0.5B model achieves competitive memory management at a fraction of the cost of RL on 7B models.]

### 6.5 Qualitative Analysis
[TBD — Sample successful and failed transitions. Where does generative rewrite help most? Where does it fail?]

### 6.6 Limitations
- **SFT ceiling**: Our method is fundamentally limited by the quality of the training data. If the GPT-4o-generated M′ targets are suboptimal, the model cannot exceed them. RL-based methods (AgeMem) can in principle discover better strategies through exploration.
- **Synthetic training data**: Our dataset is LLM-generated, not derived from real human conversations. Distribution shift between synthetic triplets and real-world memory transitions may degrade performance.
- **Model scale**: At 0.5B parameters, our model's capacity is limited. We do not claim that a 0.5B model can outperform 7B+ models in absolute terms — our claim is about cost-efficiency.
- **Benchmark scope**: We evaluate on two conversational memory benchmarks. Agent-task benchmarks (ALFWorld, SciWorld) used by AgeMem are not included, limiting direct comparison.
- **Single-backbone**: We only evaluate on Qwen 2.5 0.5B. Results may differ with other SLM architectures.

---

## 7. Conclusion

[TBD — Write after results are final. Should restate: (1) generative implicit transition is a viable alternative to discrete action selection, (2) SFT on small models achieves competitive results at low cost, (3) the approach is strongest for consolidation and abstraction where the output is genuinely generative rather than a single-record edit.]

---

## Appendix A — Dataset Statistics
[TBD — Per-category distributions, length statistics, generation prompts]

## Appendix B — Baseline Implementation Details
[TBD — Configurations for each baseline, reproduction notes for A-MEM and AgeMem]

## Appendix C — Full Per-Category Results
[TBD — Detailed breakdown of all metrics per category]

## Appendix D — Prompt Templates
[Include the 5 category-specific prompt templates from src/data_generation/prompts/]

---

*Last updated: 2026-08-01. This skeleton will be filled in as experimental results become available.*
*See TABLE.md for the detailed table plan.*