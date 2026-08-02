# Paper Summaries

Three closest competing papers, read and summarized for Phase 1 (Related Work).

---

## 1. AgeMem (ACL 2026 SAC Highlight)

**Paper:** "Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents"
**Authors:** Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, Libing Wu (Wuhan University + Alibaba)
**arXiv:** 2601.01885
**PDF:** `agemem_2601.01885.pdf`
**Code:** https://github.com/y1y5/AgeMem (✅ public, built on Trinity-RFT + Agentscope)

### Mechanism
- **Unified LTM+STM management** via tool-based actions: ADD, UPDATE, DELETE (LTM); RETRIEVE, SUMMARY, FILTER (STM)
- Agent learns to autonomously decide *when* and *which* memory operation to apply
- **Discrete action selection** — the agent picks from a fixed set of 6 memory operations (critically different from our generative approach)

### Training: 3-Stage Progressive RL with Step-wise GRPO
- **Stage 1 (LTM construction):** Agent stores useful info during casual conversation
- **Stage 2 (Distraction):** Distracting content introduced, agent must use STM tools to filter
- **Stage 3 (Integration):** Task requires coordinated use of LTM + STM
- **RL algorithm:** GRPO (Group Relative Policy Optimization) with step-wise credit assignment
- **Backbone:** Qwen2.5-7B-Instruct, Qwen3-4B-Instruct
- **Training cost:** RL fine-tuning over Trinity-RFT framework — substantial GPU requirement

### Key Results
| Backbone | Avg Score (5 benchmarks) | vs No-Memory | vs Best Baseline |
|---|---|---|---|
| Qwen2.5-7B | 41.96% | +49.6% | +4.82 pp (vs Mem0) |
| Qwen3-4B | 54.31% | +23.5% | +8.57 pp (vs A-MEM) |

- RL contributes +8.5–8.7 pp over noRL variant
- Memory Quality (MQ) on HotpotQA: 0.533 (Qwen2.5-7B), 0.605 (Qwen3-4B)
- Token reduction: 3.1–5.1% vs RAG baseline

### Benchmarks Used
ALFWorld, SciWorld, PDDL, BabyAI, HotpotQA (NOT LoCoMo or LongMemEval)

### Relevance to Our Project
- **Closest competing work** — same goal (learned memory management) but:
  - **Discrete actions** vs our **generative rewrite** (Contribution 1)
  - **RL (GRPO)** vs our **SFT** (Contribution 2 — RQ4)
  - **7B/4B models** vs our **0.5B SLM** (Contribution 2)
- **Benchmarks differ** — AgeMem uses agent-task benchmarks (ALFWorld, SciWorld, etc.), we use conversational memory benchmarks (LoCoMo, LongMemEval). Direct number comparison may not be possible.

### Reproducibility Assessment
- ✅ Code is public (GitHub: y1y5/AgeMem)
- ⚠️ Built on Trinity-RFT + Agentscope — complex dependency chain
- ❌ Training requires RL infrastructure (GRPO) — not feasible within our timeline
- **Verdict:** REPORTED-NUMBERS-ONLY for our paper (per WORKPLAN.md Day 6 checkpoint)

---

## 2. A-MEM (NeurIPS 2025)

**Paper:** "A-MEM: Agentic Memory for LLM Agents"
**Authors:** Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang (Rutgers University)
**arXiv:** 2502.12110
**PDF:** `amem_2502.12110.pdf`
**Code:** https://github.com/WujiangXu/A-mem (✅ public), https://github.com/WujiangXu/AgenticMemory (eval benchmark)

### Mechanism
- **Zettelkasten-inspired** dynamic memory organization
- **Note construction:** Each new memory → comprehensive note with structured attributes (context, keywords, tags, embedding)
- **Link generation:** Analyzes historical memories, establishes connections based on semantic similarity
- **Memory evolution:** New memories can trigger updates to existing memories' contextual representations
- **Supersede detection:** Identifies when new info makes old memories obsolete
- **No explicit training** — uses LLM (GPT-4o-mini) for note construction, linking, and evolution decisions at inference time

### Training
- **No training required** — purely heuristic + LLM-prompting based
- Uses GPT-4o-mini as the memory manager (via API calls)
- This is a key difference from both AgeMem (RL) and our approach (SFT)

### Key Results (LoCoMo)
| Model | Method | Avg F1 | Avg BLEU-1 |
|---|---|---|---|
| GPT-4o-mini | A-MEM | Best across categories | Best across categories |
| Various (6 models) | A-MEM | Superior vs MemGPT, LoCoMo baseline | — |

### Benchmarks Used
LoCoMo, DialSim

### Relevance to Our Project
- **Directly uses LoCoMo** — we can compare numbers directly
- Uses discrete detection/linking, not generative rewrite → our Contribution 1 positioning
- No training required → our Contribution 2 (SFT as lightweight alternative) is still valid
- **Reproducible** — code is public, uses GPT-4o-mini API (no training needed)

### Reproducibility Assessment
- ✅ Code is public (GitHub: WujiangXu/A-mem)
- ✅ No training required — uses LLM API calls
- ✅ Can reproduce on LoCoMo with GPT-4o-mini
- **Verdict:** REPRODUCIBLE (Baseline 5)

---

## 3. LightMem (ACL 2026)

**Paper:** "Lightweight LLM Agent Memory with Small Language Models"
**Authors:** Jiaquan Zhang, Chaoning Zhang, Shuxu Chen, et al. (UESTC + Kyung Hee + CityU HK + Oxford)
**arXiv:** 2604.07798
**PDF:** `lightmem_2604.07798.pdf`
**Code:** https://github.com/zjunlp/LightMem (✅ public)

### Mechanism
- **SLM-driven modular memory:** 3 specialized SLMs for online operations
  - Controller (SLM-1): Intent routing + HQ-based query construction
  - Selector (SLM-2): Metadata-constrained prefiltering + semantic consistency re-ranking
  - Writer (SLM-3): Incremental MTM writing
- **Offline consolidation:** Large-context LLM processes incremental batches, abstracts episodic evidence into LTM
- **3-tier memory:** STM (immediate context) → MTM (interaction summaries) → LTM (consolidated knowledge)
- **Online/offline decoupling:** Online path stays lightweight; heavy abstraction deferred to offline

### Training
- **SLMs are NOT trained** — they are pre-trained models used for structured decision tasks
- Offline consolidation uses a large LLM (not trained)
- This is a **system architecture** contribution, not a learned memory transition

### Key Results (LoCoMo, GPT-4o-mini)
| Method | Single-hop F1 | Multi-hop F1 | Temporal F1 | Open-domain F1 | Adversarial F1 | Token Length |
|---|---|---|---|---|---|---|
| LoCoMo baseline | 40.36 | 25.02 | 18.41 | 12.04 | 69.23 | 16,910 |
| A-MEM | 44.65 | 27.02 | 45.85 | 12.14 | 50.03 | 2,520 |
| **LightMem** | **45.81** | **28.85** | **46.28** | **13.52** | **54.57** | **1,150** |

- ~2.5 F1 improvement over A-MEM on LoCoMo
- 117× token reduction vs full-context
- 83ms retrieval latency, 581ms end-to-end

### Relevance to Our Project
- Shares our **SLM-driven** motivation — but uses SLMs for routing/filtering, not for learning memory transitions
- Consolidation is **LLM-based** (offline), not learned via SFT
- Directly comparable numbers on LoCoMo
- Our approach differs: we train an SLM to **directly generate M→M'** rather than routing to an LLM consolidator

### Reproducibility Assessment
- ✅ Code is public (GitHub: zjunlp/LightMem)
- ⚠️ Requires multiple SLM deployments + LLM for offline consolidation
- ✅ Can potentially run on LoCoMo
- **Verdict:** REPRODUCIBLE (but not a baseline — this is a system comparison, not a learned transition method)

---

## Positioning Summary

| Dimension | AgeMem | A-MEM | LightMem | **MSTM (Ours)** |
|---|---|---|---|---|
| **Memory operation** | Discrete action (6 tools) | Discrete detection + linking | SLM-routed, LLM-consolidated | **Generative rewrite** |
| **Training method** | RL (3-stage GRPO) | None (heuristic + LLM) | None (system design) | **SFT** |
| **Model size** | 7B/4B | GPT-4o-mini (API) | Multiple SLMs + LLM | **0.5B (local)** |
| **Training cost** | High (RL on 7B) | Zero (API calls) | Zero (API calls) | **Low (SFT on 0.5B)** |
| **Benchmarks** | ALFWorld, SciWorld, etc. | LoCoMo, DialSim | LoCoMo, DialSim | LoCoMo, LongMemEval |
| **Code available** | ✅ | ✅ | ✅ | ✅ (ours) |