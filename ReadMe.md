# FraudOps-Bench

A pilot benchmark for evaluating whether LLM agents can emulate fraud analyst investigation workflows using public fraud datasets and synthetic case files.

## Current version

FraudOps-Bench v0 contains:
- IEEE-CIS based 50-case pilot sample
- Synthetic fraud analyst SOP
- Dataframe-backed investigation tools
- Hidden fraud labels for evaluation
- Agent-readable case JSONL format

## Project structure

```text
data/
  raw/          # not committed
  processed/    # small benchmark files only
src/
  tools.py
prompts/
  sop_v0.md
notebooks/
outputs/        # not committed