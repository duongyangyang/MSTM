# Baseline 6 — AgeMem

Reproduction/adaptation of AgeMem (RL-trained unified LTM/STM memory management, ACL 2026 SAC Highlight).

## Paper
- **Title:** "Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents"
- **Authors:** Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, Libing Wu
- **arXiv:** 2601.01885
- **PDF:** `papers/agemem_2601.01885.pdf`
- **Code:** https://github.com/y1y5/AgeMem

## Mechanism Summary
- Tool-based discrete memory actions: ADD, UPDATE, DELETE (LTM); RETRIEVE, SUMMARY, FILTER (STM)
- 3-stage progressive RL training with step-wise GRPO
- Backbone: Qwen2.5-7B-Instruct, Qwen3-4B-Instruct
- Training framework: Trinity-RFT + Agentscope

## Reproducibility Verdict

**REPORTED-NUMBERS-ONLY** (not reproducible within our timeline)

### Reasons:
- [x] Code is public (✅ checked: https://github.com/y1y5/AgeMem)
- [x] Requires Trinity-RFT framework + Agentscope — complex dependency chain
- [x] RL training (3-stage GRPO) requires substantial GPU resources — infeasible for 1-2 week sprint
- [x] Trained on 7B model — our project uses 0.5B, making direct reproduction impossible
- [x] Uses different benchmarks (ALFWorld, SciWorld, PDDL, BabyAI, HotpotQA) — not LoCoMo/LongMemEval

### Action:
- Pull reported numbers from the paper (Table 2) for TABLE.md Table A3
- Mark clearly as "reported, not reproduced" in the paper
- Note benchmark mismatch: AgeMem results are on agent-task benchmarks, not conversational memory benchmarks — direct comparison may not be meaningful

## Key Numbers (from paper, Table 2)
| Backbone | ALFWorld | SciWorld | PDDL | BabyAI | HotpotQA | Average |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 41.07 | 35.55 | 17.31 | 61.42 | 54.44 | 41.96 |
| Qwen3-4B | 48.97 | 59.48 | 35.07 | 72.56 | 55.49 | 54.31 |

Status: ✅ evaluation complete (reported-numbers-only).