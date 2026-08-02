# experiments/logs/

One entry per training or evaluation run. Required before moving to the next run
(see CLAUDE.md "Conventions" — prevents re-running duplicate experiments).

Suggested format: one markdown or JSON file per run, named `YYYY-MM-DD_<method>_<benchmark>.md`, containing:
- Config used (or path to configs/ file)
- Git commit hash (if applicable)
- Key results
- Notes / anomalies
