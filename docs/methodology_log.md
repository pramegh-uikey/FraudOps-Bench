# FraudOps-Bench methodology log

Running record of what's been built, why, and what it currently shows. Kept
in sync with the codebase as work happens, in dated entries below. Intended
to seed the eventual paper's methods section, not to replace it.

See also: [`literature_survey/deep-research-report.md`](../literature_survey/deep-research-report.md)
for the prior-art/novelty analysis this project's design is based on, and
[`ReadMe.md`](../ReadMe.md) for the plain repo-structure overview.

## Current state (as of 2026-08-18)

**What FraudOps-Bench is:** a pilot benchmark testing whether LLM agents can
emulate a fraud analyst's card-not-present transaction review workflow,
built on IEEE-CIS Kaggle fraud data plus a synthetic analyst SOP and a
tool-grounded investigation environment.

**Arms compared** (all read the same 6-tool evidence surface where
applicable):

| Arm | What it is | Backend |
|---|---|---|
| `classical_ml` | HistGradientBoosting / LogisticRegression on leakage-free tabular features | scikit-learn, local |
| `direct_control` | Sees only the visible alert summary, no tool evidence, no investigation | Claude Sonnet 5 |
| `linear_api` | Single-shot: full evidence dump from all 6 tools, one completion | Claude Sonnet 5 |
| `agentic_api` | Real LangGraph agent: brain/orchestrator/tools/memory/supervisor nodes, iterative tool calls, self-checks required checks before finalizing | Claude Sonnet 5, LangGraph |
| `linear_local` / `agentic_local` | Same as above, local Ollama models (≤14B) | Ollama -- **built but not run this phase; local-model runs are explicitly paused per standing instruction** |

**Data splits** (disjoint `transaction_id`s, verified zero overlap):

| Split | n | Purpose |
|---|---|---|
| `dev` | 50 (25/25 fraud) | Original pilot set. Already spent on iterative prompt/SOP tuning during initial development -- free to keep using for further dev-time tinkering, but never conflate its numbers with holdout numbers. |
| `calibration` | 120 (60/60 fraud) | Used only to pick each arm's escalate-probability band via k-fold CV, and to catch pipeline bugs, before anything touches holdout. |
| `holdout` (v1) | 300 (150/150 fraud) | Run 2026-08-16. **Superseded and no longer authoritative** -- its `calibrate_band()`-selected bands overfit small-sample noise (see 2026-08-16 log entries). Its labels were used to diagnose and validate the fix, which spends it as a clean single-use set. Kept on disk for the record, not for reporting. |
| `holdout_v2` | 300 (150/150 fraud), fresh sample, disjoint from all prior splits | **Current, authoritative final reporting set. Run 2026-08-18** under the corrected `calibrate_band()` (see log entry below). |

**Selective prediction / escalation:** every arm outputs a calibrated
`fraud_probability`; disposition (`APPROVE`/`ESCALATE`/`REJECT`) is a
deterministic function of that probability and a per-arm band, not a
free-form model choice. Bands are picked via 5-fold CV on the calibration
split (`calibrate_band()` in `src/selective_prediction.py`), targeting
<=15% error on decided cases. `direct_control` is deliberately excluded from
band calibration and self-consistency -- it has no case-specific evidence,
so it correctly converges on "no signal" (~0.5) almost every time, and
calibrating a band against that would be calibrating noise.

**Methodology freeze:** `src/freeze_methodology.py` hashes the files that
define an arm's behavior (SOP, `configs/models.yaml`, `flows.py`,
`agentic_graph.py`, `selective_prediction.py`) into
`configs/frozen_manifest.json`. The holdout run refuses to execute unless
the current files match the frozen hashes -- this is the mechanism that
closes the original leakage problem (tuning the SOP/band by watching
accuracy on the same cases being reported).

**holdout_v2 results (n=300, current, authoritative, single-use)** -- see
the 2026-08-18 log entry below for full numbers. Headline finding, now
validated on a second, independent holdout set: `agentic_api`'s accuracy
claim genuinely holds up under a properly-calibrated band (96.1%, its best
result yet) -- but the honest cost of that accuracy is escalating 83% of
the queue. `classical_ml` remains the strongest *practical* performer
(91.0% accuracy at 85.3% coverage) because it resolves far more of the
queue confidently. The original holdout (v1) run is superseded -- its
`agentic_api` number (75.9%) was an artifact of an overfit band, not a real
capability regression; see the 2026-08-16 root-cause entry.

**Total spend:** ~$295 across the full session ($35.86 calibration-v1 +
$93.33 holdout-v1 [superseded] + $92.84 holdout_v2 + $72.85 calibration
repeat runs [variance estimate] + $0.47 LLM-judge faithfulness sample;
`classical_ml` is free/local). Balance is effectively exhausted again as of
2026-08-18.

## Open items (tracked, not all urgent)

- [x] **Holdout run** -- v1 done 2026-08-16 ($93.33), superseded. **holdout_v2 done 2026-08-18 ($92.84), current and authoritative.** See log entries below. Single-use -- do not rerun or re-tune against it.
- [x] **calibrate_band() overfitting fix** -- diagnosed and fixed 2026-08-16, validated end-to-end on holdout_v2 2026-08-18. See both log entries.
- [x] **Faithfulness / evidence-attribution scoring** -- built (`src/faithfulness.py`, `src/score_faithfulness.py`). Deterministic, zero-API-cost numeric cross-referencing against real tool evidence. ~99-99.3% verified rate for both `linear_api` and `agentic_api` across all four splits run so far (dev, calibration, holdout, holdout_v2).
- [x] **LLM-judge faithfulness Phase 2** -- built (`src/judge_faithfulness.py`, Haiku, ~25-case samples). Two real bugs found and fixed during validation before trusting it at scale (see 2026-08-18 log entry) -- both were the judge being shown incomplete evidence, not the model actually fabricating anything. Found genuine (if minor) semantic misattributions on both arms once fixed.
- [x] **Repeated-run variance estimate** -- 2 repeats each (rep1/rep2) on the calibration split for all 3 API arms, done 2026-08-18. `linear_api` 89.9%±1.7pt, `agentic_api` 95.7%±1.7pt across 3 runs (base + 2 repeats, all re-evaluated under the same current band). `direct_control` variance isn't meaningful to measure this way (near-zero decidable cases per run by design).
- [ ] Human fraud-analyst baseline -- not started, not a hard requirement for now.
- [ ] SOP practitioner validation -- SOP is self-authored, never checked against an actual analyst's judgment.
- [ ] Second dataset beyond IEEE-CIS -- deferred; candidates identified, see below.
- [ ] Paper draft / related-work writeup -- not started.

### Candidate second datasets (research only, no integration yet)

Cross-checked the existing literature survey against a fresh web search
(2026-08-14) -- nothing new has displaced IEEE-CIS as the richest public
option; it remains the primary transaction-level benchmark with interpretable
(non-PCA'd) features. For a future second dataset, ranked by fit with the
current tool-based architecture (needs interpretable categorical/behavioral
fields the 6 tools can expose, not black-box PCA components):

| Dataset | Fit | Why |
|---|---|---|
| **Sparkov / simulated CC transactions** | Good | Synthetic but rich categorical fields (merchant, category, location); used by the closest direct prior-art paper (FAA framework). Full case simulation. |
| **Fraud ecommerce** (~150k transactions) | Good | Has signup time, purchase time, device/browser/IP -- maps cleanly onto our device-history/velocity-style checks. |
| **BAF (Bank Account Fraud)** | Good, different alert type | Synthetic-from-real, privacy-preserving, six variants. Would extend the benchmark to account-opening fraud rather than just card-not-present, which is a genuinely different investigation shape. |
| **PaySim** (mobile money, ~23M synthetic) | Moderate | Large-scale, but thinner on device/identity fields than Sparkov or Fraud ecommerce. |
| **BankSim** | Moderate | Merchant-customer network features, but older and smaller (594k rows, 7.2k fraud). |
| Credit Card Fraud (ULB), Elliptic/Elliptic++, DGraphFin, AMLworld/AMLSim/SynthAML | Poor fit for now | PCA-anonymized (ULB) or crypto/graph-structured (Elliptic family, DGraphFin) or AML-alert-centric (AMLworld/AMLSim/SynthAML) -- none expose the kind of interpretable per-entity history fields the current 6 tools are built around. Would need a different tool layer, not a drop-in swap. |

Full source table: [`literature_survey/deep-research-report.md`](../literature_survey/deep-research-report.md#data-tools-and-benchmark-landscape).

---

## Log

### 2026-08-14 -- Calibration split run, leakage-fix infrastructure verified end-to-end

Built and verified (previous entry's plan): `generate_benchmark_cases.py`,
`splits.py`, `freeze_methodology.py`, `--split`-aware versions of
`generate_evidence_packets.py`/`run_baseline.py`/`run_classical_baseline.py`/
`parse_outputs.py`/`evaluate_baseline.py`/`compare_baselines.py`, and
`load_frozen_band()`/`save_frozen_band()` in `selective_prediction.py`.

Ran `classical_ml`, `direct_control`, `linear_api`, `agentic_api` on the
n=120 calibration split (explicitly not the local/Ollama arms, per standing
instruction). Total cost $35.86.

**Bug caught and fixed by the calibration split doing its job:**
`linear_api` truncated 8/120 responses exactly at the `max_tokens` cap
(6144) -- the tightened SOP's `check_verdicts` + evidence-citation
requirements made responses long enough to hit the ceiling on the more
detail-heavy cases. Bumped `max_tokens` to 8192 in `configs/models.yaml`,
retried the 8 cases, re-froze the methodology manifest afterward so the
frozen state reflects the fix.

**Band calibration (5-fold CV, target <=15% error on decided cases):**

| Arm | Band | Coverage | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| `linear_api` | (0.35, 0.65) | 57% (68/120) | 0.912 | 0.889 | 0.976 | 0.930 |
| `agentic_api` | (0.45, 0.55) | 99% (119/120) | 0.857 | 0.821 | 0.917 | 0.866 |
| `classical_ml` | (0.45, 0.55) | 95% (114/120) | 0.860 | 0.831 | 0.915 | 0.871 |
| `direct_control` | (0.4, 0.6) default, not calibrated | 1% (1/120) | n/a | -- | -- | -- |

Coverage-weighted correctness (fraction of the *full* 120-case queue
resolved correctly, not just accuracy among decided cases) tells a different
story than accuracy alone: `agentic_api` ~85.0%, `classical_ml` ~81.7%,
`linear_api` ~51.7%. `linear_api`'s single-shot design needed a much wider
band to hit the accuracy target, meaning it escalates nearly half the queue
-- high precision/recall when it commits, but it commits far less often.
`agentic_api` is not clearly beating `classical_ml` on this split -- a
finding worth reporting honestly rather than downplaying.

This is calibration-split data, explicitly not the reportable holdout
result. n=120 95% CI is roughly +/-6pts at this accuracy level; holdout at
n=300 tightens that to roughly +/-4pts.

**Budget:** original balance $65, spent $35.86, ~$29 remains. Holdout
(n=300) estimated ~$94 -- paused until topped up, per explicit user
decision to wait rather than shrink the holdout set or switch to a cheaper
model (Haiku was considered and rejected: swapping models mid-project would
add a second confound on top of the sample-size fix and break comparability
with the Sonnet-based dev-set numbers).

**Also fixed this session (not caught by calibration, caught mid-work):** a
`git checkout -- prompts/sop_v0.md` run to undo a one-line test edit
reverted the entire file to the last commit, wiping the session's SOP
rewrite (base-rate-neglect fix, calibration scale, worked examples). Caught
immediately, recovered verbatim from this session's own transcript, and
restored -- confirmed via `git diff` matching the expected shape again.
Noting it here as a process lesson: `git checkout -- <file>` on an
uncommitted file discards the *entire* working-tree diff, not just the most
recent edit.

### 2026-08-14 (later same day) -- Faithfulness / evidence-attribution scoring built

Added `src/faithfulness.py` + `src/score_faithfulness.py` (`--arm
{linear_api|agentic_api} --split {dev|calibration|holdout}`), plus a
Faithfulness section in `compare_baselines.py`. Pure post-hoc analysis --
no prompt/schema change, no API calls, works on data already collected.

**Ground truth sources confirmed before building:** `linear_api` evidence
comes from the split's `evidence_packets.jsonl` (`tool_evidence` dict,
already keyed by tool); `agentic_api` evidence comes from each run's saved
`tool_trace` (`{tool_name, step, output}`, `output` a JSON string of the
same dict shape the tools return -- verified this parses cleanly before
relying on it).

**Approach:** deterministic numeric cross-referencing (extract numeric
claims from `check_verdicts`/`evidence_used`/`risk_indicators`/
`protective_indicators`/`final_case_note`, check each against every numeric
leaf value -- and its percentage-form variant -- flattened from the real
tool evidence), plus a stricter exact-match check for citations that follow
the `tool.field = value` dotted style the SOP's worked examples encourage.
No LLM judge; that stays a documented, optional, ~$1-2 Phase 2.

**Bugs found and fixed while validating on the dev split (n=50, free,
already collected -- exactly why validation happened there first):** the
first pass showed a suspiciously high ~4% "unverified/fabrication" rate,
which on inspection was almost entirely extraction noise, not real model
errors:
- Identity-flag field names (`id_15/16/28/29`, `id_35-38`) partially
  matched as bare numbers.
- Browser/OS version strings (`chrome 63.0`, `rv:58.0`, `Trident/7.0`)
  matched as numeric claims even though they're string fields, not numeric
  ones, in the ground truth.
- Screen resolutions (`1600x900`) split into two false numeric claims.
- Percentage ranges (`5.3%-7.8%`) had the range hyphen misread as a minus
  sign on the second number, producing a claim with no positive-value match.

Fixed all four (see `_strip_non_evidentiary_patterns`, `_IDENTITY_FLAG_RE`,
`_BROWSER_OS_VERSION_RE`, `_RESOLUTION_RE`, and the `_NUMBER_RE` lookbehind
in `src/faithfulness.py`). Verified rate went from ~96% to ~99.2-99.3% on
dev after the fixes, and manual review of every remaining "unverified" case
confirmed they're a real, accepted, documented limitation, not further
bugs: the model correctly doing arithmetic the ground-truth flattener
doesn't replicate (e.g. citing "~90 seconds" between two prior
transactions, computed by subtracting two `TransactionDT` values that are
each individually present in the evidence, or "~3x" from two percentage
figures) -- derived facts, not fabricated ones.

**Calibration-split results (n=120, both arms):**

| Arm | Mean verified rate | Cases w/ any unverified claim | Uninformative citations | Citations of uncalled tools |
|---|---|---|---|---|
| `linear_api` | 0.991 | 24/120 | 165 | n/a |
| `agentic_api` | 0.991 | 28/120 | 174 | 0/120 |

Both arms are highly faithful to the evidence they cite -- essentially no
fabrication once extraction noise is removed. The more interesting finding
is **uninformative citations**: both arms fairly often name a tool in
`evidence_used` without stating what it returned (e.g. "get_card_history
output"), which is a real weakness (a citation that doesn't cite anything
checkable) distinct from fabrication, and worth reducing via a future SOP
tweak requiring every `evidence_used` entry to include a concrete value.
`agentic_api` never cited a tool it hadn't actually called (0/120) --
it doesn't fabricate investigation it didn't do.

### 2026-08-16 -- Holdout run (n=300, final, single-use)

Balance was topped up to ~$130. Re-verified the frozen manifest matched
current files (it did -- nothing methodology-related changed since the
2026-08-14 freeze; the faithfulness-scoring work added that day doesn't
touch any hashed file) before running.

**Process notes, for anyone rerunning this:**
- First launch attempt used `caffeinate -i PYTHONPATH=src python3 ...` --
  wrong argument order. `caffeinate` tried to exec a program literally
  named `PYTHONPATH=src`, failed instantly, and piping through `tee` masked
  it as exit code 0 with zero cost incurred. Correct order is
  `PYTHONPATH=src caffeinate -i python3 ...` (env var as a normal shell
  prefix before `caffeinate`, not passed as its argument). Relaunched
  correctly and verified real progress (rows actually appearing in the
  output files) before trusting it, per the same lesson from the
  calibration run's silent-failure incident.
- `caffeinate -i` was used specifically to prevent the multi-hour
  machine-sleep stall that inflated the calibration run's wall-clock time
  to ~21 hours. With it, all three arms completed in the expected range
  (no multi-hour gaps in the logs).
- One case (`linear_api`/HOLD_0293) truncated at the `max_tokens` cap
  (8192) on the first attempt -- the same failure mode as calibration, now
  down to 1/300 (0.33%) instead of 8/120 (6.7%). Retried under the
  *unchanged* frozen config rather than bumping `max_tokens` again --
  changing methodology mid-holdout-run would defeat the point of freezing
  it. Succeeded on retry (4081 tokens, well under the cap -- output length
  varies between draws of the same prompt). If a future holdout retry
  doesn't succeed, the honest thing is to report it as a parse failure /
  reduced coverage, not to loosen the cap after the fact.

**Cost:** `direct_control` $13.87, `linear_api` $38.78, `agentic_api`
$40.68 -- $93.33 total, zero errors across all 900 calls (300 cases x 3
arms). Combined with the $35.86 calibration spend, total project spend is
$129.19 of the $130 balance.

**Results (n=300, frozen bands from the calibration split, unchanged):**

| Arm | Band | Coverage | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| `linear_api` | (0.35, 0.65) | 50.7% (152/300) | 0.908 | 0.869 | 0.961 | 0.912 |
| `agentic_api` | (0.45, 0.55) | 99.7% (299/300) | 0.759 | 0.762 | 0.752 | 0.757 |
| `classical_ml` | (0.45, 0.55) | 97.0% (291/300) | 0.873 | 0.827 | 0.946 | 0.883 |
| `direct_control` | (0.4, 0.6) default | 0% (0/300) | n/a | -- | -- | -- |

95% CI half-widths at these n/accuracy pairs: `linear_api` ~+/-4.6pt,
`agentic_api` ~+/-4.9pt, `classical_ml` ~+/-3.8pt -- these are now the
genuinely defensible numbers the whole calibration/holdout split was built
to produce.

Coverage-weighted correctness (fraction of the full 300-case queue resolved
correctly, same framing used for the calibration split): `classical_ml`
84.7%, `agentic_api` 75.6%, `linear_api` 46.0%.

**Headline finding -- the calibration-split near-tie did not hold.** On
calibration, `agentic_api` (coverage-weighted ~85.0%) was slightly ahead of
`classical_ml` (~81.7%). On the actual holdout, that flips clearly:
`classical_ml` (84.7%) meaningfully outperforms `agentic_api` (75.6%) on
both raw accuracy (87.3% vs 75.9%) and coverage-weighted correctness.
`agentic_api`'s accuracy alone dropped nearly 10 points between splits
(85.7% -> 75.9%) while `classical_ml` held essentially steady (86.0% ->
87.3%). `linear_api` again shows the same pattern as calibration -- highest
per-decision accuracy (90.8%) but by far the lowest coverage (50.7%),
so it resolves less than half the queue confidently.

This is exactly the failure mode the calibration/holdout split exists to
catch: reporting the calibration-split numbers as final would have
overstated how competitive the agentic LLM arm is with classical ML. A
plausible explanation (not yet verified) is that `calibrate_band()`'s
"narrowest band that just clears the CV accuracy target" selection is a
mild multiple-comparison-style optimism, more so for `agentic_api` given
it converges tightly around 0.5 with almost no escalation (1/120 on
calibration) -- worth a closer look before over-interpreting, but the
result stands as reported: it is the number the frozen methodology
actually produced on genuinely unseen data.

**Faithfulness held stable across splits despite the accuracy drop** --
`agentic_api` verified_rate 0.992 on holdout (0.991 calibration, 0.993
dev), `linear_api` 0.990 (0.991 calibration, 0.992 dev). The accuracy
regression is not explained by evidence fabrication; `agentic_api` is
citing real evidence just as faithfully as before, so the drop likely sits
in how it weighs/interprets that evidence into a final probability, or in
the calibrated band, not in what it retrieves.

**Budget:** effectively exhausted ($0.81 left of $130). No further LLM runs
possible without topping up again.

### 2026-08-16 (later same day) -- Root-caused the agentic_api regression, fixed calibrate_band()

Investigated why `agentic_api` dropped 85.7% -> 75.9% between splits. Pure
local analysis, zero API calls -- everything needed was already on disk.

**Root cause, confirmed with data, not guessed:** `agentic_api`'s
`fraud_probability` outputs are heavily discretized -- they cluster on
round 0.05-increment values rather than being continuous (0.45 alone was
53/300 holdout cases, 0.55 was 45/300). The calibrated band `(0.45, 0.55)`
sits exactly on the two most common output values. On the 120-case
calibration split, only 12 cases landed at exactly 0.45 and 13 at exactly
0.55 -- and by sampling luck, those small groups skewed hard toward one
class (75% non-fraud at 0.45, 61.5% fraud at 0.55), making the narrow band
look excellent. On the 300-case holdout, the same exact values are close to
coin flips (47.2% fraud at 0.45, 55.6% at 0.55) -- consistent with what the
SOP itself tells the model these values mean ("0.35-0.65 is genuinely mixed
evidence"). Confirmed this is a band-selection artifact, not a deeper
capability problem, by recomputing accuracy on the *existing* holdout
probabilities under wider bands (diagnostic only, not a re-report): (0.40,
0.60) -> 79.3% accuracy at 92.0% coverage; (0.35, 0.65), the SOP's own
"mixed" range -> 94.4% accuracy at 41.7% coverage.

**Fix implemented in `calibrate_band()` (`src/selective_prediction.py`):**
band selection now uses a one-sided lower-confidence-bound on CV accuracy
(`mean - 1.645 * standard_error` across folds, `confidence_z` param) instead
of the raw mean, and skips any candidate band with fewer than
`min_decided_per_band` (default 30) total decided cases across folds
outright, regardless of how good its accuracy looks. Validated directly
against the real calibration data before and after:

| Arm | Old selection (point estimate) | New selection (LCB + min-sample) |
|---|---|---|
| `linear_api` | (0.35, 0.65) | (0.35, 0.65) -- unchanged, already had adequate support |
| `agentic_api` | (0.45, 0.55), LCB was never checked | (0.25, 0.75) -- LCB 0.895 vs the old band's LCB of only 0.794 |
| `classical_ml` | (0.45, 0.55) | (0.35, 0.65) |

Then validated end-to-end by applying the *new* bands to the *already-collected*
holdout probabilities (diagnostic, existing data, zero cost): `agentic_api`
under (0.25, 0.75) would have scored 100% accuracy at 17.7% coverage (very
conservative -- escalates over 4 in 5 cases) instead of 75.9% at 99.7%
coverage; `classical_ml` under (0.35, 0.65) would have scored 89.3% at
87.3% coverage (modest improvement over 87.3%/97.0%); `linear_api`
unchanged since its band didn't move.

**Deliberately did NOT overwrite `configs/calibrated_bands.json` or
retroactively change the reported holdout numbers.** The 75.9% figure for
`agentic_api` stands as the honest result of the methodology that was
actually frozen before that run -- swapping in a band chosen by looking at
holdout accuracy (even to validate a fix) would itself be exactly the kind
of post-hoc peeking the calibration/holdout split exists to prevent. The
fixed `calibrate_band()` is ready for the next calibration round, on fresh
data, whenever there's budget for one -- it's a methodology change and
would need a fresh freeze before any new holdout run.

### 2026-08-18 -- Validated the fix on a fresh holdout_v2, repeat-run variance, LLM-judge faithfulness

Balance topped up to ~$165. Full round: apply the fixed `calibrate_band()`
for real, validate it on a genuinely fresh (never-peeked) holdout set,
measure run-to-run variance, and build the deferred LLM-judge faithfulness
pass.

**Code changes:**
- Added a 4th split, `holdout_v2`, across `splits.py` and every
  `--split`-aware script (`run_baseline.py`, `run_classical_baseline.py`,
  `evaluate_baseline.py`, `parse_outputs.py`, `compare_baselines.py`,
  `generate_evidence_packets.py`, `score_faithfulness.py`). The original
  `holdout` (v1) is kept on disk, unmodified, for the record -- not deleted,
  not reused. `run_baseline.py`'s frozen-methodology guard was widened to
  cover both `holdout` and `holdout_v2` (it previously only checked the
  literal string `"holdout"`).
- `generate_benchmark_cases.py` gained a `--holdout-v2` mode
  (`generate_holdout_v2()`) that samples a fresh, disjoint 150/150 set via
  `load_excluded_transaction_ids()` (which now unions all of dev +
  calibration + holdout automatically) with a new random seed (303).
  Verified zero `transaction_id` overlap with all three prior splits before
  using it.
- Backed up the pre-fix `configs/calibrated_bands.json` to
  `calibrated_bands_v1_2026-08-14.json` before overwriting it with the
  LCB-selected bands, so the exact bands behind the (superseded) v1 holdout
  report stay recoverable.
- Retrained `classical_ml` excluding all 4 splits (dev+calibration+holdout+holdout_v2,
  770 transactions) -- the earlier training run only excluded 3 splits and
  would have leaked holdout_v2 rows into the training set otherwise.
- Re-froze the methodology (`configs/frozen_manifest.json`) -- the hash for
  `selective_prediction.py` correctly changed, confirming the manifest
  captures the `calibrate_band()` fix, not just a stale copy.

**Incident: Anthropic API key was invalid for the first launch of every job
in this round.** All 6 initial background runs (holdout_v2 x3, calibration
rep1 x3) failed 100% of their calls with `401 authentication_error`. Zero
cost was incurred (errors, not successful calls), and `run_baseline.py`'s
resumability logic doesn't count error rows as complete, so nothing needed
manual cleanup -- relaunching the identical commands after the user updated
`ANTHROPIC_API_KEY` in `.env` correctly retried every case from scratch.
Verified the new key with a one-word, sub-cent test call before relaunching
anything at scale. Lesson: a `wc -l > 0` check on an output file is not
suficient evidence a background run is healthy -- error rows get written
too. Now checking `error is None` on top of row counts before trusting a
launch.

**holdout_v2 results (n=300, current authoritative bands):**

| Arm | Band | Coverage | Accuracy | Precision | Recall | F1 | Coverage-weighted |
|---|---|---|---|---|---|---|---|
| `linear_api` | (0.35, 0.65) | 50.7% | 0.914 | 0.917 | 0.946 | 0.931 | 46.3% |
| `agentic_api` | (0.25, 0.75) | 17.0% | 0.961 | 0.944 | 1.000 | 0.971 | 16.3% |
| `classical_ml` | (0.35, 0.65) | 85.3% | 0.910 | 0.876 | 0.971 | 0.921 | 77.7% |
| `direct_control` | (0.4, 0.6) default | 0% | n/a | -- | -- | -- | -- |

One parse failure (`HOLD2_0237`, same `max_tokens`-cap truncation pattern,
1/300 = 0.33%, in line with holdout v1's rate). Retried once under the
unchanged frozen config per the established discipline (no `max_tokens`
change mid-holdout) -- succeeded on retry (6933 tokens, comfortably under
the 8192 cap).

**This is the validation the whole exercise was for.** `agentic_api`'s
accuracy claim holds up on genuinely fresh data -- 96.1%, its best result
across any split -- confirming the original problem really was the band,
not a fluke of the first holdout set. But the honest cost is now visible:
under the corrected, non-overfit band, `agentic_api` escalates 83% of the
queue (249/300) to hit that accuracy. `classical_ml` remains the strongest
*practical* performer by a wide margin (77.7% vs 16.3% coverage-weighted)
because it resolves far more of the queue confidently at comparable
accuracy. `linear_api` sits in between, as in every prior split.

**Repeat-run variance (calibration split, 3 runs each: base + rep1 + rep2,
all evaluated under the current band):**

| Arm | Run accuracies | Mean | Std |
|---|---|---|---|
| `linear_api` | 0.912, 0.905, 0.880 | 0.899 | ±0.017 |
| `agentic_api` | 0.938, 0.969, 0.966 | 0.957 | ±0.017 |
| `direct_control` | n/a (0-1 decidable cases per run) | -- | not meaningful |

Run-to-run variance from LLM stochasticity alone is small (~±1.7 points for
both arms) -- much smaller than the sampling-noise CI at n=120 (~±6pt) or
even n=300 (~±4pt). Single runs are not being meaningfully misled by
generation-time randomness; the earlier band-overfitting problem was a
sampling/selection issue, not a stochasticity issue.

**LLM-judge faithfulness (`src/judge_faithfulness.py`, Haiku, 25-case
samples per arm on holdout_v2):** built to catch semantic misattribution
the numeric cross-check can't -- a claim that cites a real number but
asserts something false about it. Two real bugs found and fixed *before*
trusting it at scale, both self-inflicted, neither a real model failure:

1. First version JSON-serialized the full ground-truth evidence and
   hard-truncated at 8000 characters. This silently cut off later tools
   (e.g. `device_history`, `velocity_summary`) before the judge ever saw
   them, so it flagged real, verifiable claims as fabricated purely because
   of where they landed in the truncated text. Caught by manually
   inspecting a 2-case smoke test before running the real sample --
   `get_device_history` had, in fact, been called, its output parsed fine,
   and it contained the *exact* number ("6 confirmed fraud, 30% rate") the
   judge had just called unverifiable.
2. Second version fixed that by stripping the bulky `recent_transactions`/
   `recent_device_transactions` arrays instead of truncating -- but the
   model's own prompt *includes* those arrays (see `tools.py`'s
   `get_card_history`/`get_device_history`), and it legitimately cites
   specific past-transaction details from them. Stripping them made the
   judge blind to real evidence the model actually had. Final fix: show
   the judge the complete, untruncated evidence -- Haiku's context window
   handles it easily (a few thousand tokens), so there was never a real
   reason to trim it. Cost stayed trivial either way (~$0.006-0.011/case).

Results after both fixes: `linear_api` 6/24 sampled cases had >=1 genuine
misattribution, `agentic_api` 8/24. Real examples caught: citing an
`identity_flags` field (`id_12`) that doesn't exist in that case's evidence;
asserting a "consistent device fingerprint" between two transactions when
the current transaction's `DeviceInfo` field is actually null, not matching;
misattributing a transaction reference to the wrong tool (`device_history`
vs `card_history`). One flagged item was borderline/pedantic (a rounding
difference, 0.60 vs 0.603) rather than a real error -- a known limitation
of an LLM judge worth keeping in mind when reading this metric, not a
reason to discard it. Total judge cost: $0.47 for both samples combined.

**Cost for this round:** holdout_v2 $92.84 (direct_control $14.09, linear_api
$39.10, agentic_api $39.66) + calibration repeats $72.85 (rep1 $36.43,
rep2 $36.42) + judge faithfulness $0.47 = **$166.16**. Session-to-date
total across both rounds: ~$295. Balance effectively exhausted again.

### 2026-08-19 -- Locked publish-readiness plan: self-consistency ablation, stats rigor, paper draft

Per explicit instruction: no human baseline, no second dataset, testing
capped at 48 hours from execution start, everything after that is writing
and stats only. Full plan approved and executed within budget.

**Code:**
- Added `--override-self-consistency {true,false}` to `run_baseline.py` --
  force-overrides `configs/models.yaml`'s per-arm setting for ablation
  runs, no separate config file needed.
- New `src/stats_analysis.py`: McNemar's exact test (paired correctness
  comparison, verified against the textbook (b=10, c=2) reference case --
  reproduced p=0.0386 exactly), percentile bootstrap CIs (10,000
  resamples, case-level, varying both coverage and accuracy together),
  and risk-coverage curves + AURC computed directly on `holdout_v2`'s real
  probabilities. Added `scipy` and `matplotlib` to `requirements.txt`.
- New `docs/error_taxonomy.md`: 7 named categories from the 14 LLM-judge
  findings plus 2 more from manual review (derived-arithmetic claims,
  evidence-grounded-but-wrong dispositions), with counts and traceable
  examples.
- New `paper/draft.md`: full manuscript draft (abstract through
  conclusion), every number traceable to a file on disk, no invented
  numbers.

**Self-consistency ablation** (`linear_api` + `agentic_api`, same 300
`holdout_v2` cases, same frozen bands, `--run-tag no_sc`): $61.87, zero
errors (one transient truncation on `linear_api`, same base rate as
before, retried successfully under the unchanged config). **Result: no
detectable effect.** McNemar's test on commonly-decided cases found zero
discordant pairs for both arms (p=1.0) -- self-consistency changed the
correctness of not a single shared decided case. Coverage and accuracy
differences (<2pt) are within the previously measured run-to-run noise.
A clean negative result, reported directly rather than omitted; also
confirms Section 5's calibration-band fix, not self-consistency, explains
the reported accuracy differences.

**Statistical rigor pass (zero new API cost, on existing `holdout_v2`
data):**
- Bootstrap 95% CIs confirm the point estimates: `linear_api` 0.915
  [0.868, 0.957], `agentic_api` 0.961 [0.900, 1.000], `classical_ml` 0.910
  [0.873, 0.944].
- **Risk-coverage curve figure is the strongest new result of this
  round**: `classical_ml`'s curve dominates both LLM arms across nearly
  the entire coverage range, not just at the specific chosen operating
  points -- AURC 0.0343 vs. 0.1080 (`linear_api`) vs. 0.1117
  (`agentic_api`), lower is better. Saved to
  `outputs/holdout_v2/risk_coverage_curves.png`.
- Pairwise McNemar's tests between arms are underpowered given how
  different their coverage profiles are (e.g. `agentic_api` at 17% vs.
  `classical_ml` at 85% share few commonly-decided cases) -- disclosed
  explicitly in the stats report rather than presented as a clean null
  result.
- Verified the risk-coverage curve reproduces the exact already-reported
  operating-point numbers at each arm's actual chosen band, as a
  consistency check before trusting the rest of the sweep.

**Total for this final round:** $61.87 (ablation only -- everything else
was zero-cost analysis). Project total across all rounds: ~$357.

**Status: the empirical + statistical work for a workshop-tier submission
is done.** `paper/draft.md` is a complete draft manuscript. Remaining work
is entirely writing polish (the draft still has a few `[Author note: ...]`
scaffolding comments marking where synthesis/prose needs a final human or
LLM writing pass) and a venue decision -- no further runs are planned or
needed under this locked scope.

**Supplementary data point (not in the paper draft, kept here for
reference):** `agentic_api` at the common (0.35, 0.65) band -- the same
band selected for `linear_api` and `classical_ml` -- computed from the
already-collected `holdout_v2` risk-coverage sweep, zero additional API
cost:

| Band | Coverage | Accuracy |
|---|---|---|
| (0.45, 0.55) -- original, overfit | 99.7% | 75.9% |
| (0.35, 0.65) -- common band, illustrative only | 39.7% | 92.4% |
| (0.25, 0.75) -- algorithm-selected, primary reported result | 17.0% | 96.1% |

Sits exactly where the risk-coverage curve implies it should: monotonic
between the two endpoints. Legitimate to compute (it's a read-off of the
already-swept descriptive curve, not a new band selection), but explicitly
kept out of the paper draft per instruction -- the frozen (0.25, 0.75)
result remains the single primary number for `agentic_api`.

### 2026-08-28 -- GPT-5.6 Terra integration: backend added, dev-split truncation pilot clean

Added a second model, GPT-5.6 Terra (OpenAI), as two new arms --
`linear_gpt` / `agentic_gpt` -- run through the identical pipeline, SOP,
and calibration methodology already used for `linear_api`/`agentic_api`.
Explicit goal: check whether the calibration-discretization finding and
the classical-ML-dominates-in-practice finding are Claude-specific or
general, without giving GPT any comparison advantage (own calibration,
own band, same SOP/tools/splits/stats pipeline).

**API verification before writing any code, all against real calls, not
docs/assumptions:** confirmed the live model ID (`gpt-5.6-terra`, exact
match to the marketing name), that `temperature`/`top_p` are both
REJECTED (400) same as Claude, that the request param is
`max_completion_tokens` not `max_tokens`, and pricing ($2/M input, $12/M
output, plus a real cached-input discount -- $0.20/M reads, $2.50/M
writes). One deliberately-triggered finding: a reasoning-tier model can
spend its entire token budget on invisible reasoning and return empty
visible content with `finish_reason="length"` and no error -- confirmed
live with a 20-token cap (content="", reasoning_tokens=20/20). Guarded
against explicitly in `call_openai()`, not left to surface as a generic
empty-response bug.

**Real, load-bearing discovery mid-implementation:** `bind_tools()` on
`gpt-5.6-terra` 400s over the default Chat Completions endpoint with
reasoning active ("Function tools with reasoning_effort are not
supported... use /v1/responses or set reasoning_effort to 'none'").
Setting `reasoning_effort='none'` would mean `agentic_gpt` runs without
reasoning while `linear_gpt` (plain Chat Completions, no tools) reasons
normally -- an apples-to-apples violation within GPT's own two arms.
Fixed by switching the agentic flow to the Responses API
(`use_responses_api=True`), confirmed working for tool binding,
structured output, and a full multi-turn tool-call round trip.

**Closed an integrity gap while here:** `freeze_methodology.py`'s
`METHODOLOGY_FILES` never included `llm_backends.py` -- the file where
every backend's actual call/retry/parsing logic lives, including the new
`call_openai()`. A silent edit there between calibration and holdout
would previously have gone undetected. Now hashed alongside the other 5
files.

**Dev-split truncation pilot (n=50 per arm, same methodology used to
derive Claude's `max_tokens: 8192`):** both arms clean at
`max_tokens: 16384` -- 0/50 truncations for either. Headroom was
comfortable, not borderline: `linear_gpt`'s largest case used 4,251 of
16,384 tokens (output + reasoning combined); `agentic_gpt`'s largest
single-node output was 4,721. No adjustment needed.

**Incidental finding from the pilot, not yet a conclusion (n=50, not the
real calibration/holdout sample):** `agentic_gpt` averaged $0.1576/case
vs. `agentic_api`'s $0.132/case (~19% more expensive on this sample),
despite a real 47% cache-hit rate on input tokens (`cached_input_tokens`
averaged 23,156 of 49,742 average input tokens). `agentic_gpt`'s average
input-token count (49,742) also ran notably higher than `agentic_api`'s
(28,731 on the real holdout_v2 run) -- plausibly the Responses API's
encrypted reasoning-trace blocks getting resent as part of conversation
history across the graph's multiple sequential LLM calls per case, though
this hasn't been confirmed by inspecting the raw message payloads. Worth
running down before reporting cost numbers, not assumed.

Next: calibration run (n=120) for both arms, `calibrate_arm.py` (new
reusable driver -- `calibrate_band()`/`save_frozen_band()` previously had
no CLI anywhere in `src/`, only `notebooks/exp1.ipynb`), refreeze, then
`holdout_v2`.

### 2026-08-27 -- GPT-5.6 Terra calibration split run (n=120) and band selection

Ran `linear_gpt`/`agentic_gpt` on the n=120 calibration split (own
from-scratch calibration, per the apples-to-apples requirement --
deliberately never reused `linear_api`/`agentic_api`'s frozen bands). Both
runs got killed partway by what was almost certainly clamshell sleep
(`caffeinate -dimsu` was active but can't override a closed lid without an
external display) -- `linear_gpt` at 94/120, `agentic_gpt` at 35/120, zero
corrupted/error rows in either partial file. Resumed both via
`run_baseline.py`'s existing dedup/resume logic (reloads existing output,
skips already-completed case_ids) with no data loss. Both finished clean:
120/120, 0 errors, 0 parse failures.

**Band selection (`calibrate_arm.py`, same LCB methodology as
`linear_api`/`agentic_api`):**

| Arm | Band | LCB acc. | Mean acc. | Total decided (of 120) |
|---|---|---|---|---|
| `linear_gpt` | (0.2, 0.8) | 0.884 | 0.931 | 39 |
| `agentic_gpt` | (0.1, 0.9) -- fallback | 1.0 | 1.0 | 12 |

`linear_gpt` selected organically -- the half_width=0.30 candidate cleared
both the 85% LCB-accuracy target and the >=30-decided-case floor.

`agentic_gpt` hit `calibrate_band()`'s fallback path -- worth stating
plainly, not smoothing over. No candidate band cleared both bars: the
closest, half_width=0.30, landed at LCB 0.8494 (just under the 0.85
target) with 34 decided cases; the next step up, half_width=0.35, hit a
perfect LCB 1.0 but only 21 decided cases, and was correctly rejected by
the >=30-decided-case guard as "a suspiciously perfect score from too few
cases is not evidence" (exactly the docstring's own stated rationale for
that guard existing). Fell through to the widest fallback band, (0.1,
0.9), which itself only has 12 decided cases in calibration -- LCB=1.0
there is trivial on so few points and should not be over-trusted either.
This is a genuine, unforced result of applying the identical corrected
methodology to a new model, not a bug or a parameter chosen to produce a
particular outcome; the guard did exactly what it's supposed to do
(refuse false confidence, degrade to maximally conservative rather than
silently accept a narrow band on thin support). Flagging as a real
possibility worth watching on `holdout_v2`: this band implies `agentic_gpt`
may escalate a very large fraction of the holdout queue.

Next: refreeze (`models.yaml`, `agentic_graph.py`, `flows.py`,
`llm_backends.py`, `selective_prediction.py` all changed since the last
freeze), then the single-use `holdout_v2` run for both arms.
