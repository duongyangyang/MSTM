# Prompt template — Contradiction Resolution

You are generating training data for a memory state transition model. Your task is to create realistic (M, ΔM, M′) triplets where new information contradicts an existing memory record and the agent must resolve the conflict.

## Category: Contradiction Resolution

**Definition:** New information (ΔM) directly conflicts with one or more records in the current memory (M). The agent must resolve the conflict by keeping the newer/correct fact and annotating the old one as outdated using inline annotations.

**Key constraint:** M′ should be roughly the same length as M. Update only contradicted records — keep unchanged facts exactly as-is. Use concise inline "(previously: X)" annotations. Do NOT add separate "previously" bullet points.

**Sub-types to cover (mix evenly):**
1. **Temporal contradiction** — same fact, different time periods (e.g., "I live in Tokyo" vs "I moved to Osaka last month")
2. **Factual contradiction** — two statements that cannot both be true (e.g., "I'm allergic to peanuts" vs "I ate peanut butter yesterday")
3. **Preference flip** — user explicitly states a change in preference that contradicts a strongly-stated earlier preference
4. **Identity/role contradiction** — conflicting information about the user's role, expertise, or identity
5. **Source-reliability contradiction** — the user corrects a previously inferred (possibly wrong) memory

## Output Format

Generate exactly ONE valid JSON object per line (JSONL format). Each object must have this structure:

```json
{
  "M": "Current memory state as concise bullet points. At least one record should be contradicted by delta_M.",
  "delta_M": "The new information that contradicts M. Should make the contradiction obvious.",
  "M_prime": "The evolved memory state after resolving the contradiction. Conflicting records are resolved with inline annotations. M_prime should be roughly the same length as M — update only contradicted facts, keep others exactly as-is.",
  "category": "contradiction_resolution"
}
```

## Quality Requirements

1. **Genuine contradiction:** The conflict between M and ΔM should be clear and non-trivial.
2. **Reasoned resolution:** M_prime should show the agent has thought about which fact to trust.
3. **Compact output:** M_prime should be roughly the same length as M. Use inline "(previously: X)" annotations. Do NOT add separate "previously" bullet points — they bloat the output.
4. **Temporal annotation where appropriate:** When the contradiction is temporal, include time markers inline (e.g., "since 2024", "graduated 2024").
5. **Implicit operation:** Do NOT include explicit labels like [CONTRADICT] or [RESOLVED] in the output.
6. **Diverse personas:** Vary the user persona across examples.
7. **Language:** Generate all content in English.

## Examples

### Example 1 — Temporal Contradiction
```json
{
  "M": "- User is a PhD student at MIT in computer science\n- Research focus: reinforcement learning\n- Lives in Cambridge, MA\n- Has a dog named Apollo\n- Runs marathons as a hobby",
  "delta_M": "Actually, I graduated from my PhD last spring. I'm now a research scientist at DeepMind in London.",
  "M_prime": "- User is a research scientist at DeepMind in London (graduated PhD from MIT, spring 2024)\n- PhD in CS from MIT (focus: reinforcement learning)\n- Lives in London, UK (previously: Cambridge, MA)\n- Has a dog named Apollo\n- Runs marathons as a hobby",
  "category": "contradiction_resolution"
}
```

### Example 2 — Factual Contradiction
```json
{
  "M": "- User is lactose intolerant\n- Avoids all dairy products\n- Prefers plant-based milk alternatives\n- Enjoys cooking Asian cuisine",
  "delta_M": "I had the most amazing cheese fondue last night. I've actually been eating dairy for a while now — turns out I grew out of the lactose intolerance.",
  "M_prime": "- User is NOT lactose intolerant (outgrew it; previously: avoided all dairy)\n- Enjoys dairy (e.g., cheese fondue)\n- Prefers plant-based milk alternatives (taste preference, not medical)\n- Enjoys cooking Asian cuisine",
  "category": "contradiction_resolution"
}
```

### Example 3 — Preference Flip
```json
{
  "M": "- User strongly dislikes remote work and prefers in-office collaboration\n- Believes remote work reduces team productivity\n- Currently works hybrid (3 days in office, 2 days remote)\n- Works as a product manager at a fintech startup",
  "delta_M": "I've completely changed my mind about remote work. After our team went fully remote for 3 months during the office renovation, our productivity actually went up. I'm never going back to an office.",
  "M_prime": "- User strongly prefers remote work (previously: preferred in-office; changed after 3-month remote trial)\n- Now believes remote work improves productivity (previously: believed it reduced it)\n- Works fully remotely as a PM at a fintech startup\n- Previously worked hybrid (3 office / 2 remote)",
  "category": "contradiction_resolution"
}
```

## Generation Instructions

Generate 100 diverse examples for the "contradiction_resolution" category. Each example on a separate line. Vary:
- The type of contradiction (temporal, factual, preference, identity, source-reliability)
- The number of memory records in M (3–8)
- The user persona
- The complexity of the resolution (single-fact vs. multi-fact cascading resolution)

**IMPORTANT:** M_prime should be roughly the same length as M, not longer. Update only contradicted facts; keep unchanged facts exactly as-is. Use inline "(previously: X)" annotations instead of adding separate "previously" bullet points.

Output ONLY valid JSONL — one JSON object per line, no markdown fences, no commentary.