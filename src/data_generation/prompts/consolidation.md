# Prompt template — Consolidation

You are generating training data for a memory state transition model. Your task is to create realistic (M, ΔM, M′) triplets where new information allows merging multiple related memory records into a more compact, coherent representation.

## Category: Consolidation

**Definition:** New information (ΔM) is related to existing memory records in M. Rather than simply appending the new fact, the agent should consolidate the related records into fewer, more informative records. The evolved memory (M′) is more compact than M but preserves all critical information.

**Key constraint:** M′ must be MORE COMPACT than M — fewer characters, fewer bullet points. Merge related facts into concise records. Do NOT expand or add narrative prose.

**Sub-types to cover (mix evenly):**
1. **Skill/knowledge consolidation** — multiple related skills or learning experiences are merged into a higher-level summary
2. **Experience consolidation** — repeated similar experiences are compressed into a pattern or summary
3. **Relationship consolidation** — multiple facts about the same person/entity are merged
4. **Project consolidation** — multiple updates about the same project are combined into a current-status summary
5. **Health/medical consolidation** — related health information is consolidated into a coherent profile

## Output Format

Generate exactly ONE valid JSON object per line (JSONL format). Each object must have this structure:

```json
{
  "M": "Current memory state containing multiple related but separate records on the same topic. Should have some redundancy or fragmentation that consolidation can fix.",
  "delta_M": "New information that is related to the fragmented records in M. This is the trigger for consolidation.",
  "M_prime": "The evolved memory state after consolidation. Related records are merged into fewer, richer records. M_prime MUST be shorter than M (fewer characters, fewer bullet points). Use concise bullet points, no narrative prose.",
  "category": "consolidation"
}
```

## Quality Requirements

1. **Genuine redundancy:** M should have real fragmentation that consolidation meaningfully improves. Don't just merge two unrelated records.
2. **Information preservation:** M_prime must not lose any critical facts from M. The consolidation should make the memory more useful, not just shorter.
3. **Compact output:** M_prime MUST be shorter than M. Merge related records into a single concise bullet. No narrative prose — keep the same bullet-point style as M.
4. **Natural trigger:** ΔM should be the natural reason why consolidation happens now — it's the new piece that makes the pattern visible.
5. **Implicit operation:** Do NOT include explicit labels like [CONSOLIDATE] or [MERGED] in the output.
6. **Diverse domains:** Vary the topic area — skills, health, relationships, projects, travel, etc.
7. **Language:** Generate all content in English.

## Examples

### Example 1 — Skill Consolidation
```json
{
  "M": "- User learned Python basics through an online course\n- User completed a PyTorch tutorial and built a simple image classifier\n- User has been practicing data preprocessing with pandas\n- User works as a business analyst at a logistics company\n- Lives in Toronto, Canada",
  "delta_M": "I just finished the Deep Learning Specialization on Coursera and built a transformer-based sentiment analysis model from scratch.",
  "M_prime": "- User has ML engineering skills: Python, PyTorch, pandas, transformers (Deep Learning Specialization, built sentiment analysis model + image classifier)\n- Works as a business analyst at a logistics company\n- Lives in Toronto, Canada",
  "category": "consolidation"
}
```

### Example 2 — Experience Consolidation
```json
{
  "M": "- User visited Tokyo in March 2024 and enjoyed the food\n- User visited Kyoto in April 2024 and loved the temples\n- User visited Osaka in May 2024 for a business conference\n- User is learning Japanese (beginner level)\n- User works in international sales",
  "delta_M": "Just got back from my fourth trip to Japan — spent two weeks in Hokkaido hiking and visiting onsens. I think Japan is becoming my second home at this point.",
  "M_prime": "- User has been to Japan 4 times: Tokyo, Kyoto, Osaka, Hokkaido (considers it a second home)\n- Learning Japanese (beginner)\n- Works in international sales",
  "category": "consolidation"
}
```

### Example 3 — Health Consolidation
```json
{
  "M": "- User has been experiencing headaches 2-3 times per week\n- User reported difficulty sleeping and waking up tired\n- User started a new high-stress job 3 months ago\n- User drinks 4-5 cups of coffee daily\n- User exercises once a week (running)",
  "delta_M": "My doctor says I have tension headaches and early signs of burnout. She recommended reducing caffeine, improving sleep hygiene, and exercising more regularly.",
  "M_prime": "- User has tension headaches and early burnout (doctor diagnosed): symptoms include frequent headaches, poor sleep\n- Contributing factors: high-stress job (3 months), high caffeine (4-5 cups/day), low exercise (1x/week)\n- Doctor's advice: reduce caffeine, improve sleep, exercise more",
  "category": "consolidation"
}
```

## Generation Instructions

Generate 100 diverse examples for the "consolidation" category. Each example on a separate line. Vary:
- The domain (skills, health, travel, relationships, projects, etc.)
- The number of related records to be consolidated (2–5)
- The user persona
- The complexity of the consolidation (simple merge vs. restructured representation)

**CRITICAL:** M_prime must be visibly shorter than M. Count the characters — if M has 200 chars across 4 bullets, M_prime should have ~120 chars across 2-3 bullets. Use concise bullet points. No narrative prose.

{domain_instruction}

Output ONLY valid JSONL — one JSON object per line, no markdown fences, no commentary.