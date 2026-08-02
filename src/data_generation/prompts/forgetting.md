# Prompt template — Forgetting

You are generating training data for a memory state transition model. Your task is to create realistic (M, ΔM, M′) triplets where new information makes some existing memory records obsolete or low-value, and those records should be forgotten (removed).

## Category: Forgetting

**Definition:** New information (ΔM) reveals that some records in M are no longer useful — they are outdated, trivial, redundant, or irrelevant to future interactions. The evolved memory (M′) should remove these low-value records. This is NOT the same as update (where a fact is replaced) — forgetting is about removing information that has lost its utility entirely.

**Key constraint:** M′ must be SHORTER than M. Remove obsolete records entirely. For completed events, keep only a concise summary if it has lasting value.

**Sub-types to cover (mix evenly):**
1. **Completed/expired items** — one-time events, completed tasks, resolved issues that won't be referenced again
2. **Obsolete preferences** — preferences tied to a specific context that no longer applies
3. **Trivial/transient information** — low-value details that were never worth keeping long-term
4. **Superseded context** — background information that was only relevant to a past situation
5. **Redundant specifics** — overly detailed records where a summary suffices

## Output Format

Generate exactly ONE valid JSON object per line (JSONL format). Each object must have this structure:

```json
{
  "M": "Current memory state containing some records that are useful and some that should be forgotten. The mix should be realistic — not all records should be forgettable.",
  "delta_M": "New information that reveals which records are now obsolete. This is the trigger for forgetting.",
  "M_prime": "The evolved memory state after forgetting. Obsolete records are removed. M_prime MUST be shorter than M (fewer bullet points, fewer characters). Keep concise summaries of completed events only if they have lasting value.",
  "category": "forgetting"
}
```

## Quality Requirements

1. **Selective forgetting:** Not all records in M should be forgotten. M_prime should still contain useful information.
2. **Clear utility judgment:** The decision to forget should be clearly motivated by ΔM.
3. **Compact output:** M_prime MUST be shorter than M. Remove obsolete records — don't add new ones. For completed events, keep only a one-line summary if needed.
4. **No information loss of important facts:** M_prime must not lose any information that would be useful for future interactions.
5. **Implicit operation:** Do NOT include explicit labels like [FORGET] or [DELETE] in the output.
6. **Diverse scenarios:** Vary why forgetting happens — completion, expiration, triviality, context shift, redundancy.
7. **Language:** Generate all content in English.

## Examples

### Example 1 — Completed/Expired Items
```json
{
  "M": "- User was planning a trip to Barcelona for August 15-22\n- User researched hotels in the Gothic Quarter with a budget of €150/night\n- User booked flights with Vueling (outbound Aug 15, return Aug 22)\n- User is a software engineer at Google\n- User has a shellfish allergy\n- User's mother lives in Seville, Spain",
  "delta_M": "I'm back from Barcelona! It was amazing — the Gothic Quarter was perfect, and I even got to visit my mom in Seville for a weekend.",
  "M_prime": "- User is a software engineer at Google\n- User has a shellfish allergy\n- User's mother lives in Seville, Spain\n- Recently visited Barcelona (Aug 2024), also visited mother in Seville",
  "category": "forgetting"
}
```

### Example 2 — Obsolete Preferences
```json
{
  "M": "- User is shopping for a new laptop for university\n- Budget: under $1,200\n- Needs: good battery life, lightweight, suitable for CS coursework\n- User is starting first year at University of Washington\n- User is majoring in computer science\n- User lives in Terry Hall dormitory",
  "delta_M": "I finally bought the MacBook Air M3! Got it with the student discount for $1,099. It's perfect for my CS classes.",
  "M_prime": "- Owns MacBook Air M3 ($1,099, student discount)\n- First-year CS student at University of Washington\n- Lives in Terry Hall dormitory",
  "category": "forgetting"
}
```

### Example 3 — Trivial/Transient Information
```json
{
  "M": "- User had a dentist appointment on Tuesday at 2pm (Dr. Patel)\n- User's favorite coffee order is an oat milk latte\n- User is working on a presentation for the quarterly review next Friday\n- User's team has 8 members: 4 engineers, 2 designers, 1 PM, 1 data analyst\n- User is a team lead at a SaaS company\n- User has been at the company for 4 years",
  "delta_M": "Dentist went fine — just a routine cleaning. The quarterly review presentation went really well too, our team's metrics were up 15%.",
  "M_prime": "- User's favorite coffee order is an oat milk latte\n- User is a team lead at a SaaS company (4 years)\n- Team has 8 members: 4 engineers, 2 designers, 1 PM, 1 data analyst\n- Recent quarterly review: team metrics up 15%\n- Dentist: Dr. Patel (routine checkups)",
  "category": "forgetting"
}
```

## Generation Instructions

Generate 100 diverse examples for the "forgetting" category. Each example on a separate line. Vary:
- The reason for forgetting (completion, expiration, triviality, context shift, redundancy)
- The proportion of records to be forgotten (20%–50% of M)
- The user persona
- The complexity of the forgetting decision (obvious vs. nuanced judgment call)

**CRITICAL:** M_prime must be visibly shorter than M. Count the bullet points — M should lose 1-3 records. For completed events, a one-line summary is enough.

Output ONLY valid JSONL — one JSON object per line, no markdown fences, no commentary.