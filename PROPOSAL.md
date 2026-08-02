# Research Proposal

## Learning Memory State Transition for Long-Term Conversational Agents

---

## 1. Introduction and Motivation

Large Language Model (LLM)-based agents are increasingly expected to maintain long-term memory across multiple conversations and extended periods of interaction. Recent memory frameworks such as retrieval-augmented memory systems, vector databases, and graph-based memories have significantly improved the ability of agents to store and retrieve information. However, most existing approaches treat memory as a collection of static records that are continuously accumulated over time.

This design introduces several challenges. As memory grows, the system accumulates outdated, redundant, conflicting, and low-value information. For example, an agent may store both "the user lives in Hanoi" and "the user moved to Ho Chi Minh City" without explicitly representing their temporal relationship. Similarly, multiple episodic memories such as "the user learned Python," "the user learned PyTorch," and "the user learned Transformers" may remain as isolated facts rather than being consolidated into a higher-level understanding of the user's expertise.

Current solutions mainly focus on improving retrieval mechanisms or introducing forgetting heuristics. While these approaches can partially mitigate memory growth, they do not address a more fundamental question:

**How should an agent's memory evolve when new information arrives?**

Inspired by human memory systems, where memories are continuously updated, consolidated, abstracted, and sometimes forgotten, this proposal introduces the concept of **Memory State Transition**, a framework that models memory as a dynamic structure rather than a static database.

---

## 2. Related Work

Long-term memory for LLM agents has become an active research area, and several recent systems overlap with parts of this proposal. This section positions our approach relative to them and identifies the specific gap we target.

| System | Core Idea | Relation to this proposal |
|---|---|---|
| **MemGPT / Letta** (Packer et al., 2023) | Manages context as an OS-style problem: core memory + archival storage, paged in/out of the context window | Establishes the "memory as managed state" framing, but memory records themselves remain largely static once stored |
| **MemoryBank** (Zhong et al., 2024) | Long-term memory with update and forgetting mechanisms explicitly inspired by human memory | Closest in *motivation* to our human-memory framing, but relies on hand-designed update/decay rules rather than a learned transition function |
| **Mem0 / Mem0g** (Chhikara et al., 2025) | Dynamically extracts, organizes, and retrieves memory using LLM-based extraction and (optionally) graph structure | A strong, widely-used production baseline; memory updates are performed by prompting a large LLM at each step rather than a dedicated trained transition model |
| **A-MEM** (Xu et al., 2025) | Introduces dynamic links between memory entries and a "supersede detection" mechanism to address staleness | Directly overlaps with our *Update* operation, but treats update as a detection/linking problem rather than a generative rewrite of the memory state |
| **Zep** (Rasmussen et al., 2025) | Temporal knowledge-graph architecture optimized for multi-session temporal reasoning | Overlaps with our temporal-reasoning research question (RQ3), but is structured around graph edits rather than a general (M, ΔM) → M' transition |
| **LightMem** (2026) | Lightweight memory system driven by Small Language Models; separates online retrieval/writing from offline consolidation | Closest in *system goal* (SLM-driven, low-overhead memory management); consolidation is handled by a large-context LLM offline, while online SLM components are largely routing/decision modules rather than a unified generative transition model |
| **AgeMem** (Yu et al., 2026; ACL 2026) | Unifies long-term and short-term memory management directly into the agent's policy; exposes memory operations (add/update/summarize/discard) as tool-based actions, trained via a three-stage progressive reinforcement learning strategy with step-wise GRPO | **Closest competing work.** Learns *when and which discrete operation* to apply to memory. Reports strong results across five long-horizon benchmarks. |

### Positioning and gap

Two observations motivate this proposal despite the crowded landscape above:

1. **Explicit action selection vs. generative state rewrite.** Systems such as A-MEM and AgeMem model memory evolution as selecting from a discrete set of operations (add, update, delete, summarize). This is a natural fit for simple updates, but is an awkward fit for *consolidation* and *abstraction*, where the correct output is not an edit to a single record but a new synthesized statement drawn from multiple records (e.g., "Python + PyTorch + Transformers" → "experience in NLP and deep learning"). We instead propose modeling the transition as a **direct generative rewrite**: T(M, ΔM) → M′, where the model produces the full updated memory state in one pass, without first committing to a discrete operation label. The operation (update, consolidation, abstraction, forgetting) is *implicit* in the generated output rather than predicted as a classification target.

2. **Training cost.** AgeMem's reinforcement learning approach (three-stage curriculum, step-wise GRPO to handle sparse and discontinuous rewards) is effective but expensive and comparatively difficult to reproduce. We propose training via **supervised fine-tuning (SFT)** on a constructed (M, ΔM, M′) triplet dataset — a simpler, cheaper, and more stable training regime. This is a deliberate trade-off: SFT is bounded by the quality of the constructed dataset and is unlikely to exceed a well-trained RL policy in the best case, but it is significantly cheaper to train and easier to reproduce, which aligns with this project's goal of a lightweight, locally-deployable memory manager.

These two points — generative implicit transition, and SFT as a lightweight alternative to RL-based memory management — are the two pillars of differentiation for this work, and are treated as the primary technical contributions (see Section 8).

---

## 3. Research Problem

Existing memory systems can be abstracted as:

```
Conversation
     ↓
Memory Extraction
     ↓
Store
     ↓
Retrieve
```

In this paradigm, memory records are largely immutable after being stored.

This proposal reformulates memory management as a state transition problem.

Given:

- Current memory state (M)
- Newly observed information (ΔM)

the objective is to learn a transition function:

**T(M, ΔM) → M′**

where M′ is an updated memory state that is more accurate, compact, and useful for future reasoning tasks. Critically, T is realized as a single generative model call that produces M′ directly — it does not first classify which of {update, consolidate, abstract, forget} applies.

### Examples

**Memory Update**

Current memory:
- User lives in Hanoi

New information:
- User moved to Ho Chi Minh City

Updated memory:
- Current city: Ho Chi Minh City
- Previously lived in Hanoi

**Memory Consolidation**

Current memory:
- User learned Python
- User learned PyTorch

New information:
- User learned Transformers

Updated memory:
- User has experience in NLP and deep learning
- Supporting experiences: Python, PyTorch, Transformers

The key challenge is to automatically learn these transformations instead of relying on manually designed rules or discrete action classifiers.

---

## 4. Research Questions

**RQ1:** Can learned memory state transitions improve long-term question-answering performance compared with static memory systems?

**RQ2:** Can memory evolution reduce memory size and redundancy while preserving critical information?

**RQ3:** Does memory evolution improve temporal reasoning and conflict resolution in long-term conversations?

**RQ4:** Compared to discrete, RL-trained memory management (AgeMem), does a generative, SFT-trained transition model achieve competitive task performance — particularly on consolidation- and abstraction-heavy scenarios — at substantially lower training cost?

---

## 5. Proposed Method

### 5.1 Memory State Transition Framework

We propose a Memory State Transition Model (MSTM) that continuously updates the memory repository whenever new information is observed.

The framework consists of four implicit memory operations:

- Update
- Consolidation
- Abstraction
- Forgetting

Instead of explicitly predicting these operations as classification labels, the model directly generates an updated memory state.

The overall pipeline is:

```
Conversation
     ↓
Memory Extraction
     ↓
Memory State Transition Model
     ↓
Updated Memory Store
     ↓
Retriever
     ↓
LLM Response
```

This design allows memory to evolve over time while maintaining consistency and compactness.

### 5.2 Lightweight Memory Manager

A central objective of this project is to design a memory management model that can run locally on commodity hardware.

Rather than using large proprietary models such as GPT-4o or Claude for memory maintenance — or training via reinforcement learning as in AgeMem — we propose fine-tuning a small language model (SLM) via supervised fine-tuning, such as:

- SmolLM
- Qwen 0.5B
- Phi-3 Mini

The model only performs memory transitions and therefore requires significantly less computational capacity than a general-purpose LLM, and significantly less training infrastructure than an RL-based policy.

This enables practical deployment in local agents and privacy-sensitive applications, and directly addresses RQ4.

---

## 6. Dataset Construction

Existing datasets primarily evaluate retrieval and reasoning rather than memory evolution.

Therefore, we propose constructing a transition dataset consisting of triplets:

**(M, ΔM, M′)**

where:

- M: current memory state
- ΔM: new information
- M′: evolved memory state

The dataset will contain several categories:

**Update**
- profile changes
- preference changes
- location changes

**Contradiction Resolution**
- conflicting user facts
- temporal updates

**Consolidation**
- multiple related memories
- repeated experiences

**Abstraction**
- episodic-to-semantic memory transitions

**Forgetting**
- obsolete or low-value information

The initial dataset may be generated using a strong LLM and subsequently refined through manual inspection.

A target size of 5,000–20,000 examples is considered sufficient for a conference-scale study.

---

## 7. Experimental Setup

### Baselines

The proposed approach will be compared against:

**Baseline 1: Static Memory**
Store all memories without modification.

**Baseline 2: Time-Decay Forgetting**
Remove memories based on age.

**Baseline 3: Heuristic Consolidation**
Merge memories using similarity thresholds.

**Baseline 4: LLM-Based Memory Management**
Use a large language model to update memories (e.g., Mem0-style prompting).

**Baseline 5: A-MEM**
Dynamic memory linking with supersede detection, representing the state of the art in explicit, detection-based memory update.

**Baseline 6: AgeMem**
RL-trained, action-based memory management, representing the state of the art in learned memory management. Used to directly evaluate RQ4 (performance and cost comparison against a generative, SFT-trained alternative).

### Evaluation Datasets

Primary evaluation benchmarks:

- LoCoMo
- LongMemEval

These benchmarks contain long-horizon conversations, temporal reasoning tasks, and memory-intensive question answering.

---

## 8. Evaluation Metrics

We adopt a **two-track evaluation** approach: (1) end-to-end QA on conversational memory benchmarks to measure downstream utility, and (2) internal transition-quality evaluation on our held-out test split to measure per-operation transition quality directly.

### Track 1: Benchmark QA (End-to-End Utility)

**Primary Metrics (Standard)**
- **LLM-Judge Accuracy** — binary correct/partial/incorrect via GPT-4o judge (standard for LoCoMo, LongMemEval, and AgeMem). This is the primary metric — F1 and EM are reported as secondary since they poorly reflect semantic correctness in open-ended generation.
- **Token-Level F1** and **Exact Match (EM)** — secondary generation metrics.
- **Memory Quality (MQ)** — AgeMem-style: LLM evaluates stored memory M′ against session-summary ground truth on a 0–1 scale. Directly measures memory state quality, not just downstream answer quality.

**Retrieval Quality**
- Recall@K, MRR (Mean Reciprocal Rank)

**Memory Efficiency**
- Number of memory records, total stored tokens, compression ratio
- Per-category breakdown by benchmark-native question categories

**Consistency Metrics**
- Contradiction rate, redundancy rate (heuristic + optional LLM-based)

**Temporal Reasoning**
- Accuracy on temporal questions, update-sensitive question accuracy

**Training Cost (supports RQ4)**
- GPU-hours to train, number of training examples required, inference latency per memory update

### Track 2: Internal Transition-Quality Eval (Per-Operation — Direct Evidence)

On our held-out test split of (M, ΔM, M′) triplets, we directly measure how well predicted M′ matches gold M′ — grouped by our 5 training operation categories. This is the **direct evidence** for the paper's central claim that generative rewrite excels at consolidation and abstraction.

- **ROUGE-L** — longest common subsequence F-measure between predicted and gold M′
- **Token F1** — standard token-level overlap
- **Fact Preservation Rate** — LLM judge: fraction of gold-M′ facts preserved in predicted M′
- **Transition Judge Score** — LLM judge: 0.0/0.5/1.0 semantic match of predicted vs gold M′ given input (M, ΔM)
- **Compression Fidelity** — predicted compression ratio vs gold compression ratio per operation type

---

## 9. Expected Contributions

This work is expected to make three primary contributions.

**Contribution 1: Generative Implicit Memory State Transition**

We introduce a formulation of long-term memory management that models memory as a dynamic evolving state rather than a static collection of records, and — in contrast to discrete, action-based approaches such as A-MEM and AgeMem — realizes this transition as a single generative rewrite in which the operation performed (update, consolidation, abstraction, forgetting) is implicit in the output rather than an explicitly predicted label.

**Contribution 2: Lightweight SFT-Based Memory Manager as an Alternative to RL-Based Management**

We propose a small language model, trained via supervised fine-tuning on constructed (M, ΔM, M′) triplets, capable of performing memory evolution locally. This offers a substantially cheaper and more reproducible training regime than reinforcement-learning-based approaches (e.g., AgeMem), trading a potentially lower performance ceiling for significantly reduced training cost and easier deployment — a trade-off we measure directly (RQ4).

**Contribution 3: Memory Evolution Dataset and Evaluation Protocol**

We construct a benchmark for studying memory transitions, including updating, consolidation, abstraction, and forgetting behaviors, along with an evaluation protocol that reports not only task accuracy but also training cost — enabling direct cost/performance comparison between generative and RL-based memory management approaches.

---

## 10. Expected Outcomes and Conclusion

We hypothesize that memory evolution is a more fundamental solution than retrieval improvements or simple forgetting heuristics, and that generative, implicitly-operating transitions are a better fit for consolidation- and abstraction-heavy scenarios than discrete action selection.

Specifically, we expect the proposed approach to:

- improve long-term QA accuracy relative to static and heuristic baselines,
- reduce memory redundancy,
- reduce storage requirements,
- improve temporal reasoning,
- maintain consistent memory representations over time,
- approach the performance of RL-trained memory managers (AgeMem) at a fraction of the training cost.

The central thesis of this research is that long-term memory systems should not merely store and retrieve information, nor simply learn to select among a fixed set of memory operations. Instead, they should continuously evolve their memory structures in response to new experiences via a directly learned, generative transition function. By learning memory state transitions in this way, conversational agents can maintain a compact, accurate, and adaptive memory that more closely resembles human memory processes — at a training cost accessible outside large industrial labs.
