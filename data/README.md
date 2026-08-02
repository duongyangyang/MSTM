# data/

- `raw/` — raw generated (M, ΔM, M′) triplets before manual review; raw LoCoMo/LongMemEval downloads go here too.
- `processed/` — cleaned, split (train/val/test) datasets ready for training and evaluation.

Schema (see CLAUDE.md "Conventions"):
```json
{"M": "...", "delta_M": "...", "M_prime": "...", "category": "update|contradiction_resolution|consolidation|abstraction|forgetting"}
```
