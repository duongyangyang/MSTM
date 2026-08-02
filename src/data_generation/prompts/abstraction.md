# Prompt template — Abstraction

You are generating training data for a memory state transition model. Your task is to create realistic (M, ΔM, M′) triplets where new information triggers the formation of a higher-level, abstracted understanding from specific episodic memories.

## Category: Abstraction

**Definition:** New information (ΔM) enables the agent to generalize from specific episodic memories to a higher-level semantic understanding. This is the most cognitively sophisticated operation: episodic-to-semantic transition. The evolved memory (M′) should replace the specific episodes with a concise abstracted record, keeping only 1-2 of the most representative episodes as grounding evidence.

**Key constraint:** M′ must be MORE COMPACT than M. The abstraction compresses multiple episodes into a single insight — the whole point is to reduce memory footprint while preserving the essential pattern.

**Sub-types to cover (mix evenly):**
1. **Trait/identity abstraction** — from specific behaviors to personality traits or identity statements
2. **Pattern recognition** — from repeated events to a recognized pattern
3. **Value/belief abstraction** — from specific decisions to underlying values or beliefs
4. **Expertise abstraction** — from specific projects to a domain expertise summary
5. **Relationship abstraction** — from specific interactions to a relationship quality summary

## Output Format

Generate exactly ONE valid JSON object per line (JSONL format). Each object must have this structure:

```json
{
  "M": "Current memory state containing specific episodic records (3-6 items). These should be concrete facts/events, not already abstracted.",
  "delta_M": "New information (another episode or fact) that, combined with M, enables abstraction.",
  "M_prime": "The evolved memory state after abstraction. Contains: (1) one concise abstracted insight that synthesizes the pattern, (2) at most 1-2 representative episodes as grounding evidence, (3) any non-episodic facts from M that are still relevant. M_prime MUST be visibly shorter than M.",
  "category": "abstraction"
}
```

## Quality Requirements

1. **Episodic-to-semantic:** The abstraction must go from concrete events to a general understanding. Don't just summarize — synthesize.
2. **Non-obvious pattern:** The abstraction should reveal something that isn't trivially stated in any single record.
3. **Compact output:** M_prime MUST be shorter than M. Replace the list of episodes with a single abstracted insight + 1-2 best examples. Do NOT preserve all original episodes.
4. **Implicit operation:** Do NOT include explicit labels like [ABSTRACT] or [PATTERN] in the output.
5. **Diverse domains:** Vary the type of abstraction — personality traits, values, expertise, preferences, relationships.
6. **Language:** Generate all content in English.

## Examples

### Example 1 — Trait/Identity Abstraction
```json
{
  "M": "- User volunteered to mentor junior developers on the team\n- User regularly stays late to help colleagues debug issues\n- User organized a weekly knowledge-sharing lunch for the engineering team\n- User declined a higher-paying offer to stay with the current team\n- User works as a senior backend engineer",
  "delta_M": "I spent my weekend helping a former colleague prepare for their job interview. I didn't even think twice about it — I just genuinely enjoy helping people grow.",
  "M_prime": "- User has a strong identity as a mentor and community-builder: consistently prioritizes helping others grow over personal gain (e.g., declined higher-paying offer, spends weekends helping former colleagues prepare for interviews)\n- User works as a senior backend engineer",
  "category": "abstraction"
}
```

### Example 2 — Pattern Recognition (Preference)
```json
{
  "M": "- User spent 3 weeks backpacking in Southeast Asia and loved the spontaneity\n- User chose a startup over a corporate job because it offered more variety\n- User gets bored with repetitive tasks and frequently asks for new challenges\n- User changed careers from accounting to UX design because accounting was too routine",
  "delta_M": "I'm quitting my job to freelance — I've lined up 3 different clients in completely different industries and I'm really excited about the variety.",
  "M_prime": "- User has a strong preference for variety and novelty: consistently chooses unpredictable, diverse experiences over stability (e.g., freelancing across industries, career change from accounting to UX)\n- User works in UX design, now freelancing",
  "category": "abstraction"
}
```

### Example 3 — Value/Belief Abstraction
```json
{
  "M": "- User donates 5% of monthly income to environmental causes\n- User switched to a fully plant-based diet for environmental reasons\n- User sold their car and now only uses public transit and bike\n- User volunteers monthly at a local river cleanup initiative",
  "delta_M": "I'm changing jobs to work at a clean energy startup. I'm taking a 20% pay cut but it's worth it to work on something that actually matters for the climate.",
  "M_prime": "- User's core value is environmental sustainability: consistently makes major life decisions prioritizing climate impact over financial or convenience considerations (e.g., took 20% pay cut for clean energy job, gave up car, adopted plant-based diet)\n- User works at a clean energy startup",
  "category": "abstraction"
}
```

## Generation Instructions

Generate 100 diverse examples for the "abstraction" category. Each example on a separate line. Vary:
- The type of abstraction (trait, pattern, value, expertise, relationship)
- The number of supporting episodes in M (3–6)
- The user persona
- The depth of the abstraction (simple pattern vs. insightful synthesis)

**CRITICAL:** M_prime must be visibly shorter than M. Count the lines — if M has 5 records, M_prime should have 2-3 records. The abstraction replaces the episode list, not supplements it.

{domain_instruction}

Output ONLY valid JSONL — one JSON object per line, no markdown fences, no commentary.