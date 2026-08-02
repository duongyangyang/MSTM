# Prompt template — Update

You are generating training data for a memory state transition model. Your task is to create realistic (M, ΔM, M′) triplets where new information updates an existing memory record.

## Category: Update

**Definition:** New information (ΔM) supersedes, corrects, or extends an existing fact about the user. The old fact is no longer accurate. The evolved memory (M′) should reflect the new truth, using inline annotations like "(promoted 2024)" or "(previously: X)" instead of adding separate historical records.

**Key constraint:** M′ should be roughly the same length as M. Update only the changed facts — keep unchanged facts exactly as-is. Use inline annotations, not extra bullet points.

**Sub-types to cover (mix evenly):**
1. **Profile changes** — name, age, job title, education, marital status
2. **Preference changes** — dietary preferences, hobbies, music/movie tastes, communication style
3. **Location changes** — city, country, workplace, travel status
4. **Relationship changes** — family status, team membership, social connections
5. **Goal/project changes** — current project, learning goals, career objectives

## Output Format

Generate exactly ONE valid JSON object per line (JSONL format). Each object must have this structure:

```json
{
  "M": "Current memory state as concise bullet points. Each record should be a self-contained factual statement.",
  "delta_M": "The new information that triggers the update. Should clearly contradict or extend a record in M.",
  "M_prime": "The evolved memory state. Old facts are updated with inline annotations. M_prime should be roughly the same length as M — update only changed facts, keep others exactly as-is. No extra 'previously' bullet points.",
  "category": "update"
}
```

## Quality Requirements

1. **Realism:** M should look like a real agent's memory — 3–8 memory records about a single user, covering diverse topics.
2. **Clear transition:** delta_M should make it obvious why M needs to change, and M_prime should show the correct resolution.
3. **Compact output:** M_prime should be roughly the same length as M. Use inline "(promoted 2024)" or "(previously: X)" annotations. Do NOT add separate "Previously..." bullet points — those bloat the output.
4. **Implicit operation:** M_prime should be generated directly — do NOT include explicit labels like [UPDATE] or [DELETE] in the output.
5. **Diverse personas:** Vary the user persona across examples — different ages, professions, life situations.
6. **Language:** Generate all content in English.

## Examples

### Example 1 — Profile Update
```json
{
  "M": "- User's name is Sarah Chen\n- Works as a junior data analyst at TechCorp\n- Lives in San Francisco\n- Enjoys hiking and photography\n- Has been learning Spanish for 6 months",
  "delta_M": "I just got promoted to senior data analyst! I'll be leading the customer insights team now.",
  "M_prime": "- User's name is Sarah Chen\n- Works as a senior data analyst at TechCorp (promoted 2024), leads customer insights team\n- Lives in San Francisco\n- Enjoys hiking and photography\n- Has been learning Spanish for 6 months",
  "category": "update"
}
```

### Example 2 — Preference Update
```json
{
  "M": "- User is a vegetarian\n- Favorite cuisine: Italian\n- Dislikes spicy food\n- Prefers home-cooked meals over restaurants",
  "delta_M": "I've actually started eating fish recently — my doctor recommended it for my iron levels. But I still don't eat other meat.",
  "M_prime": "- User is pescatarian (eats fish, no other meat; previously vegetarian)\n- Favorite cuisine: Italian\n- Dislikes spicy food\n- Prefers home-cooked meals over restaurants",
  "category": "update"
}
```

### Example 3 — Location Update
```json
{
  "M": "- User is based in London, UK\n- Works remotely for a US-based startup\n- Has a cat named Luna\n- Plays piano as a hobby",
  "delta_M": "Next month I'm relocating to Berlin for my partner's new job. The startup is fine with it since I'm already remote.",
  "M_prime": "- User is based in Berlin, Germany (relocated from London)\n- Works remotely for a US-based startup\n- Has a cat named Luna\n- Plays piano as a hobby\n- Partner works in Berlin",
  "category": "update"
}
```

## Generation Instructions

Generate 100 diverse examples for the "update" category. Each example on a separate line. Vary:
- The type of update (profile, preference, location, relationship, goal)
- The number of memory records in M (3–8)
- The user persona (age, profession, life stage, interests)
- The complexity of the update (simple single-fact change vs. multi-fact cascading update)

**IMPORTANT:** M_prime should be roughly the same length as M. Use inline annotations. Do NOT add extra bullet points just for historical context.

Output ONLY valid JSONL — one JSON object per line, no markdown fences, no commentary.