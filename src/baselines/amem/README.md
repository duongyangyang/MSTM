# Baseline 5 — A-MEM

Reproduction/adaptation of A-MEM (dynamic memory linking + supersede detection, NeurIPS 2025).

## Paper
- **Title:** "A-MEM: Agentic Memory for LLM Agents"
- **Authors:** Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang
- **arXiv:** 2502.12110
- **PDF:** `papers/amem_2502.12110.pdf`
- **Code:** https://github.com/WujiangXu/A-mem (production-ready), https://github.com/WujiangXu/AgenticMemory (eval benchmark)

## Mechanism Summary
- Zettelkasten-inspired dynamic memory organization
- Note construction: each new memory → comprehensive note with structured attributes
- Link generation: semantic similarity-based connections between memories
- Memory evolution: new memories trigger updates to existing memories
- Supersede detection: identifies obsolete memories
- No training required — uses GPT-4o-mini API for note construction, linking, and evolution

## Reproducibility Verdict

**REPRODUCIBLE** (feasible within our timeline)

### Reasons:
- [x] Code is public (✅ checked: https://github.com/WujiangXu/A-mem)
- [x] No training required — uses LLM API calls (GPT-4o-mini)
- [x] Uses LoCoMo benchmark — directly comparable to our results
- [x] Can be run with OPENAI_API_KEY and the public codebase

### Action:
- Clone repo and run on LoCoMo with GPT-4o-mini
- Record results in TABLE.md Table 1, Table A3
- Mark as "reproduced" (same hardware/data)

## Key Numbers (from paper, LoCoMo, GPT-4o-mini)
| Category | F1 | BLEU-1 |
|---|---|---|
| Average (across categories) | Best among baselines | Best among baselines |

Status: ⬜ not started (reproduction pending).