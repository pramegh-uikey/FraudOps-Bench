# FraudOps-Bench

A benchmark for evaluating whether LLM agents can emulate a fraud
analyst's card-not-present transaction investigation *process* -- not
just predict a fraud label, but gather evidence via tools, reason under
an explicit standard operating procedure (SOP), and produce an auditable
disposition with cited evidence -- built on public IEEE-CIS Kaggle fraud
data plus a synthetic analyst SOP and a tool-grounded investigation
environment.

The current empirical + statistical work is complete and written up in
[`paper/draft.md`](paper/draft.md) (headline result: an agentic LLM arm's
accuracy claim holds up on held-out data, but a classical ML baseline
remains the stronger practical performer once escalation coverage is
accounted for -- see the paper for the full calibration-failure story).
Full narrative history of every run, bug, and decision is in
[`docs/methodology_log.md`](docs/methodology_log.md).

## What's in the benchmark

Each case exposes a visible alert summary (amount, product code, card
type, purchaser/recipient email domain, device type) and, for
evidence-bearing arms, six leakage-free tools mirroring real fraud-ops
case-management tooling: `get_transaction_details`, `get_card_history`,
`get_email_domain_profile`, `get_device_history`, `get_velocity_summary`,
`get_identity_match_summary`.

**Comparison arms** (`configs/models.yaml`):

| Arm | Description |
|---|---|
| `direct_control` | Zero-evidence control -- sees only the alert summary. |
| `linear_api` | Single-shot completion given all 6 tools' output at once. |
| `agentic_api` | A LangGraph agent (brain/orchestrator/tools/memory/supervisor) that iteratively chooses which tools to call. |
| `classical_ml` | HistGradientBoostingClassifier on leakage-free tabular features (no LLM). |

`linear_local` / `agentic_local` (Ollama-backed, e.g. `qwen2.5:14b`) are
built but not part of the headline comparisons -- see the paper's
Limitations section.

**Evaluation splits** (`src/splits.py`, `docs/methodology_log.md`):

- `dev` (n=50) -- free to use for iterative prompt/SOP development.
- `calibration` (n=120) -- used only to select each arm's escalate-band
  via k-fold cross-validation.
- `holdout` (n=300) -- superseded first holdout; retained for the
  paper's before/after calibration-fix narrative.
- `holdout_v2` (n=300) -- current, authoritative single-use final
  reporting set, disjoint from every other split and from
  `classical_ml`'s training data.

A methodology-freeze mechanism (`src/freeze_methodology.py`,
`configs/frozen_manifest.json`) hashes every file that defines an arm's
behavior and refuses to run a holdout split unless the current files
match the frozen hashes -- this is what makes the paper's calibration
before/after account provable rather than just claimed.

## Project structure

```text
src/
  agentic_graph.py, flows.py, flow_types.py   # LangGraph agent + linear/direct flows
  llm_backends.py, tools.py                   # model backends, the 6 investigation tools
  run_baseline.py                             # run an LLM arm on a split
  run_classical_baseline.py                   # train/run classical_ml
  generate_benchmark_cases.py, splits.py      # dev/calibration/holdout/holdout_v2 case generation
  freeze_methodology.py                       # methodology-freeze/verify guard
  selective_prediction.py                     # calibrate_band(), self-consistency, band application
  faithfulness.py, score_faithfulness.py,     # deterministic + LLM-judge faithfulness scoring
    judge_faithfulness.py
  compare_baselines.py, evaluate_baseline.py, # cross-arm comparison, metrics
    parse_outputs.py, stats_analysis.py       # McNemar's, bootstrap CIs, risk-coverage/AURC
configs/
  models.yaml            # arm/backend definitions
  calibrated_bands.json  # current (LCB-fixed) per-arm escalate bands
  frozen_manifest.json   # methodology-freeze manifest
prompts/
  sop_v0.md              # the fraud-analyst SOP driving all LLM arms
data/
  raw/                   # not committed -- IEEE-CIS Kaggle CSVs
  processed/              # benchmark case JSONL files (committed)
docs/
  methodology_log.md     # full run-by-run history, costs, bugs, decisions
  error_taxonomy.md      # faithfulness/disposition error categorization
literature_survey/       # prior-art research pass backing paper/draft.md's Related Work
paper/
  draft.md                # the manuscript
outputs/                 # not committed -- per-arm run results, metrics, figures
```

## Running it

Requires `ANTHROPIC_API_KEY` (and `configs/models.yaml`'s Ollama arms
need a local Ollama server) in `.env`. Install dependencies from
`requirements.txt`.

```bash
# Run an LLM arm on a split
python src/run_baseline.py --arm agentic_api --split dev

# Run the classical baseline
python src/run_classical_baseline.py --split dev

# Verify the methodology hasn't drifted before running a holdout split
python src/freeze_methodology.py --verify

# Compare all arms on a split
python src/compare_baselines.py --split holdout_v2
```

See `docs/methodology_log.md` for the exact commands and costs behind
every reported number, and `paper/draft.md` for the full write-up.

## Status

Empirical and statistical work for the current submission round is done
(see `docs/methodology_log.md`'s 2026-08-19 entry). No further API runs
are planned under the current locked scope; remaining work is writing
polish and a venue decision. Explicitly deferred: a human-analyst
baseline, SOP validation by a practicing analyst, and a second dataset
beyond IEEE-CIS (candidates ranked in `docs/methodology_log.md`).
