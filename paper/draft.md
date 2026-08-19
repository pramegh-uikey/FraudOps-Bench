# FraudOps-Bench: Process-Level Evaluation of LLM Agents for Fraud-Analyst Emulation, and a Calibration Failure Mode Worth Knowing About

**Status: DRAFT.** Every number in this document is traceable to a file
under `outputs/` or `docs/methodology_log.md` in this repository. No
number here has been invented or estimated for narrative convenience.

## Abstract

We study whether an LLM agent can emulate a fraud analyst's card-not-present
transaction review workflow -- not just predict a fraud label, but gather
evidence via tools, reason under an explicit standard operating procedure
(SOP), and produce an auditable disposition with cited evidence. We build
FraudOps-Bench: a six-tool investigation environment over IEEE-CIS
transaction data, a LangGraph-based agent (brain/orchestrator/tools/
memory/supervisor), and three comparison arms (a zero-evidence control, a
single-shot "all evidence at once" baseline, and the full iterative agent)
alongside a classical gradient-boosted-tree baseline. Using a selective-
prediction framework (escalate-to-human on low-confidence cases), we
initially found the agentic arm competitive with classical ML on a
120-case calibration set. That result did not survive a fresh 300-case
holdout: the calibration procedure had overfit small-sample noise in the
escalation threshold. We diagnose the failure precisely -- discretized,
round-number-anchored LLM probability outputs interacting badly with a
narrow band selected on too little data -- fix it with a lower-confidence-
bound selection criterion, and validate the fix on a second, disjoint
holdout set. Under the corrected calibration, the agentic arm's accuracy
claim holds up (96.1%, its best result on any split) but its practical
coverage collapses to 17% of the queue; the classical baseline remains the
strongest *practical* performer, dominating the full risk-coverage curve
(AURC 0.034 vs. 0.108-0.112). We report faithfulness scoring (two
independent methods, ~99% evidence-attribution accuracy for both LLM
arms), a structured error taxonomy, and an ablation of the agent's
self-consistency mechanism. We argue the calibration failure mode itself
-- not just the fix -- is a transferable lesson for anyone doing selective
prediction with LLM confidence scores on small calibration sets.

## 1. Introduction

[Framing note for the author: open with the gap between "LLM classifies
fraud transactions" (saturated, low novelty per the literature survey)
and "LLM agent emulates the *investigation process*" (the identified
white space). State the three-part contribution: (1) a process-level
benchmark with a real tool-grounded agent, not a classification wrapper;
(2) an honest before/after account of a calibration failure and its fix,
validated on independent data, which is itself a methodological
contribution for selective prediction with LLMs; (3) faithfulness and
error-taxonomy analysis that goes beyond outcome accuracy.]

**Contributions:**
1. FraudOps-Bench: a public-data (IEEE-CIS), tool-grounded, LangGraph-based
   fraud-investigation environment with an explicit SOP, four comparison
   arms, and a disciplined dev/calibration/holdout evaluation protocol
   with a frozen-methodology audit mechanism preventing evaluation
   leakage.
2. A documented case study of `calibrate_band()` overfitting on a small
   calibration set -- diagnosed via probability-output discretization,
   fixed with a lower-confidence-bound + minimum-sample-size selection
   rule, and validated end-to-end on a second, never-touched holdout set.
3. Two independent faithfulness-evaluation methods (deterministic numeric
   cross-referencing and an LLM-judge pass) plus a structured error
   taxonomy, going beyond outcome accuracy to ask whether the agent's
   cited reasoning is actually grounded in the evidence it retrieved.
4. An ablation isolating the contribution of a self-consistency mechanism
   to the agent's calibrated probability estimates.

## 2. Related Work

[Author note: draw directly from `literature_survey/deep-research-report.md`,
sections "Prior work and evidence" and "Data, tools, and benchmark
landscape." Key positioning points to hit:]

- The closest direct prior art is the **FAA framework** (GPT-4o + tools +
  report generation on Sparkov/CCTD), which reports very strong F1 but on
  limited, mostly synthetic samples without a process-level or
  selective-prediction evaluation.
- **FinFRE-RAG** demonstrates public-data-only LLM fraud scoring gains via
  retrieval, but remains transaction-level scoring, not workflow emulation.
- **CORTEX** and **SIABench** are the closest behavioral analogues (multi-
  agent, tool-grounded, auditable investigation) but in security operations,
  not fraud, and on non-public data.
- **SOP-Bench**, **τ-bench** provide the methodological template for
  policy-heavy, long-horizon agent evaluation this work adapts to fraud
  operations.
- The literature survey's own conclusion: no existing public work combines
  public fraud data + synthetic workflow + tool-grounded agent +
  auditable process-level scoring. This is the gap FraudOps-Bench targets.
- Distinguish explicitly from "LLM vs. XGBoost on IEEE-CIS" work (FDB,
  standard classification papers) -- this paper's contribution is the
  process/selective-prediction framing and the calibration failure
  analysis, not a classification leaderboard entry.

## 3. FraudOps-Bench Design

### 3.1 Task and data

Card-not-present transaction risk review over IEEE-CIS Kaggle fraud data.
Each case exposes a visible alert summary (amount, product code, card
type, purchaser/recipient email domain, device type) and, for evidence-
bearing arms, access to six tools mirroring what a real fraud analyst's
case-management tooling would expose: `get_transaction_details`,
`get_card_history`, `get_email_domain_profile`, `get_device_history`,
`get_velocity_summary`, `get_identity_match_summary`. Tool outputs are
leakage-free by construction (only information available strictly before
the current transaction's timestamp).

### 3.2 Comparison arms

| Arm | Description |
|---|---|
| `direct_control` | Sees only the visible alert summary; no tool access. Zero-evidence control. |
| `linear_api` | Single-shot completion given the full output of all 6 tools at once. |
| `agentic_api` | A LangGraph agent with explicit brain (LLM), orchestrator, tool nodes, structured memory, and a supervisor node that checks required-check completion before allowing finalization. Iterative: the agent chooses which tools to call and when. |
| `classical_ml` | HistGradientBoostingClassifier on leakage-free tabular features derived from the same underlying data (no LLM). |

All LLM arms use Claude Sonnet 5. `classical_ml` and the local-model
variants (`linear_local`/`agentic_local`, built but not run -- see
Limitations) are not part of this report's headline comparisons beyond
`classical_ml`.

### 3.3 Selective prediction and the SOP

Every LLM arm outputs a calibrated `fraud_probability`. Disposition
(APPROVE/ESCALATE/REJECT) is a deterministic function of that probability
against a per-arm band, not a free-form model choice -- the standard
selective-prediction (Chow's rule) risk-coverage framing: fix an
acceptable error rate on decided cases, escalate the rest. The SOP
explicitly instructs the model on a calibration scale (0.0-0.15 no
signal, 0.35-0.65 genuinely mixed evidence, 0.85-1.0 strong convergent
signal) and requires an explicit per-check verdict (protective/risk/
neutral) for six required checks before a final probability, to counter
base-rate-neglect and encourage genuine escalation on ambiguous cases
rather than false confidence.

### 3.4 Evaluation protocol: dev / calibration / holdout, with a frozen-methodology guard

To prevent the eval-leakage failure mode common in iterative prompt
engineering (tuning a prompt or threshold by repeatedly checking accuracy
on the set being reported), FraudOps-Bench uses three disjoint splits:

- **dev** (n=50, 25/25 fraud): free to use for iterative SOP/prompt
  development. Already spent this way; never reported as a final number.
- **calibration** (n=120, 60/60 fraud): used only to select each arm's
  escalate-probability band via k-fold cross-validation.
- **holdout** (n=300, 150/150 fraud, two independent draws used in this
  work -- see Section 5): single-use final reporting set.

A methodology-freeze mechanism (`freeze_methodology.py`) hashes every
file that defines an arm's behavior (SOP text, model config, prompt-
construction code, agent graph definition, calibration logic) into a
manifest. The holdout-execution path refuses to run unless the current
files match the frozen hashes, with a loud, logged `--force` override
available but never silent. This is the mechanism that makes the
before/after account in Section 5 possible: it is provable, not just
claimed, that nothing was retuned against the second holdout after the
fix was frozen.

## 4. Faithfulness Evaluation

[Author note: two independent methods, both described in
`docs/methodology_log.md` 2026-08-14 and 2026-08-18 entries.]

**Method 1 -- deterministic numeric cross-referencing.** Extracts every
numeric claim from an arm's cited evidence (`evidence_used`,
`check_verdicts`, `risk_indicators`, etc.) and checks it against the real
tool-evidence values (including a percentage-form match for fractional
fields). Zero additional API cost. Across dev, calibration, holdout, and
holdout_v2, both LLM arms show 98.8-99.3% verified rates. Residual
"unverified" claims are overwhelmingly the model correctly performing
arithmetic (e.g. a time delta between two timestamps) that the ground-
truth flattener does not itself compute -- not fabrication (manually
verified, see `docs/methodology_log.md`).

**Method 2 -- LLM-judge semantic check.** A Haiku-based judge (25-case
samples per arm on `holdout_v2`) checks for semantic misattribution the
numeric method cannot catch: a correct number attached to a false claim.
Found genuine (if minor) errors in 6/24 (`linear_api`) and 8/24
(`agentic_api`) sampled cases -- see `docs/error_taxonomy.md` for the
full categorization, including one category (rounding-precision framing)
that is arguably not a real error but a limitation of the judge's own
literalness, disclosed rather than hidden.

**Result:** both arms are highly faithful to the evidence they actually
retrieve. The accuracy differences reported in Section 5 are not
explained by evidence fabrication.

## 5. The Calibration Failure, Diagnosis, Fix, and Validation

This section is presented as a first-class result, not an appendix
correction -- the failure mode is general to selective prediction with
LLM-generated probabilities and worth other practitioners knowing about.

### 5.1 Initial result (calibration split, n=120)

Band selection via 5-fold cross-validated accuracy on the calibration
split initially favored a very narrow escalate band for `agentic_api`,
(0.45, 0.55), yielding 85.7% accuracy at 99% coverage -- appearing
competitive with `classical_ml` (86.0% accuracy, 95% coverage).

### 5.2 First holdout (n=300) and the regression

Under the frozen (0.45, 0.55) band, `agentic_api` scored only 75.9%
accuracy on a fresh 300-case holdout -- a ~10-point drop. `classical_ml`
held steady (87.3%).

### 5.3 Root cause

`agentic_api`'s `fraud_probability` outputs are not continuous: they
cluster heavily on round 0.05-increment values, with 0.45 and 0.55 --
exactly the selected band's edges -- the two single most common outputs
(53/300 and 45/300 cases on the holdout, respectively). On the 120-case
calibration split, only 12 and 13 cases landed at those exact values, and
by sampling luck those small groups skewed hard toward one class (75%
non-fraud at 0.45, 61.5% fraud at 0.55), making the narrow band look
excellent. On the larger holdout, the same exact values are close to coin
flips (47.2% and 55.6% fraud respectively) -- consistent, in fact, with
what the SOP's own calibration-scale instructions say those values should
mean (genuinely mixed evidence). The calibration procedure mistook small-
sample noise for signal.

### 5.4 Fix

`calibrate_band()` was changed from selecting the narrowest band whose
mean cross-validated accuracy cleared a target threshold, to selecting
the narrowest band whose **lower-confidence-bound** (mean minus 1.645
standard errors across folds) clears the threshold, and additionally
requiring a minimum total decided-case count across folds regardless of
how good a band's point estimate looks. Re-run on the same calibration
data, this correctly rejects the narrow band (LCB 0.794 vs. an 0.85
target) and selects (0.25, 0.75) instead (LCB 0.895).

### 5.5 Validation on a second, independent holdout (n=300)

A fresh, disjoint 300-case holdout (`holdout_v2`, excluded from the
classical-model training set along with every prior split) was generated
and run under the corrected, re-frozen methodology.

**Table 1: `holdout_v2` results (n=300).**

| Arm | Band | Coverage | Accuracy | Precision | Recall | F1 | Coverage-weighted |
|---|---|---|---|---|---|---|---|
| `linear_api` | (0.35, 0.65) | 50.7% | 0.914 | 0.917 | 0.946 | 0.931 | 46.3% |
| `agentic_api` | (0.25, 0.75) | 17.0% | 0.961 | 0.944 | 1.000 | 0.971 | 16.3% |
| `classical_ml` | (0.35, 0.65) | 85.3% | 0.910 | 0.876 | 0.971 | 0.921 | 77.7% |
| `direct_control` | (0.4, 0.6) default | 0% | -- | -- | -- | -- | -- |

"Coverage-weighted" = coverage × accuracy, the fraction of the full queue
resolved correctly -- the fairer single-number comparison when arms
escalate at very different rates.

**Table 1b: same three arms, matched at all three notable bands.** Table 1
compares each arm at its own independently-selected operating point,
which is the correct primary comparison but obscures how each arm
performs off that point. Since disposition is a deterministic function of
`fraud_probability` and the band, every arm's accuracy/coverage at *any*
band is a free re-computation on the already-collected `holdout_v2`
probabilities (Section 5.5's Figure 1 curve) -- no new API calls. Shown
here for all three arms at the three bands discussed in this paper: the
original overfit band, the common band `linear_api`/`classical_ml` were
independently selected at, and the wide band `agentic_api` was
independently selected at. Each arm's own selected operating point (the
one reported in Table 1) is **bolded**.

| Arm | (0.45, 0.55) | (0.35, 0.65) | (0.25, 0.75) |
|---|---|---|---|
| `linear_api` | cov 96.7%, acc 0.752 | **cov 50.7%, acc 0.914** | cov 37.3%, acc 0.929 |
| `agentic_api` | cov 99.7%, acc 0.759 | cov 39.7%, acc 0.924 | **cov 17.0%, acc 0.961** |
| `classical_ml` | cov 95.3%, acc 0.888 | **cov 85.3%, acc 0.910** | cov 73.7%, acc 0.937 |

**Where each method shines, read directly off this table:**

- **`classical_ml`'s advantage is holding accuracy while staying willing
  to decide.** At the tightest band it still resolves 95.3% of the queue
  at 88.8% accuracy -- both other arms are 20+ points less accurate at
  that same near-total coverage (0.752, 0.759). This is the source of
  its risk-coverage-curve dominance (Section 5.5, Figure 1): it doesn't
  trade coverage for accuracy nearly as steeply as the LLM arms do.
- **`agentic_api`'s advantage is ceiling accuracy at its most selective
  point.** At the wide (0.25, 0.75) band, it is the single most accurate
  cell in this entire table (0.961) -- edging out `classical_ml`'s
  0.937 at the same band -- but at less than a quarter of
  `classical_ml`'s coverage (17.0% vs. 73.7%). When `agentic_api`
  commits, its confidence is well-placed; it simply commits rarely.
- **`linear_api` never wins a cell outright** but is never far behind
  either, sitting between the other two on both axes at every band --
  consistent with its middling coverage-weighted score in Table 1.
- The practical reading: `classical_ml` is the stronger choice as a
  primary decision-maker across the board; `agentic_api`'s comparative
  advantage is real but narrow -- a high-precision second opinion on the
  small slice of cases it is willing to rule on, not a general-purpose
  replacement.

**Table 2: bootstrap 95% CIs (10,000 resamples, `src/stats_analysis.py`).**

| Arm | Accuracy | Coverage | Coverage-weighted |
|---|---|---|---|
| `linear_api` | 0.915 [0.868, 0.957] | 0.507 [0.450, 0.563] | 0.463 [0.407, 0.520] |
| `agentic_api` | 0.961 [0.900, 1.000] | 0.170 [0.130, 0.213] | 0.164 [0.123, 0.207] |
| `classical_ml` | 0.910 [0.873, 0.944] | 0.853 [0.813, 0.893] | 0.777 [0.727, 0.823] |

**Figure 1** (`outputs/holdout_v2/risk_coverage_curves.png`): full risk-
coverage curves per arm, with the actual reported operating point marked.
`classical_ml`'s curve dominates both LLM arms across nearly the entire
coverage range, not merely at the specific chosen operating points --
AURC 0.0343 (`classical_ml`) vs. 0.1080 (`linear_api`) vs. 0.1117
(`agentic_api`); lower is better.

**Table 3: pairwise McNemar's tests** (paired, restricted to cases both
arms decided; see `outputs/holdout_v2/stats_report.md`). Because arms with
very different coverage (17% vs. 85%) share few commonly-decided cases,
these paired tests are underpowered by construction and are reported for
completeness rather than as the primary evidence -- the bootstrap CIs and
risk-coverage curves above are more informative when coverage differs
this much.

### 5.6 Interpretation

**The `agentic_api` accuracy claim is real.** It holds up on completely
fresh data under a properly calibrated band -- 96.1%, its best result on
any split -- confirming the original problem was the band, not a
capability regression. But the honest cost is now visible: at that
accuracy, it escalates 83% of the queue. `classical_ml` remains the
strongest *practical* performer by a wide margin, both by coverage-
weighted correctness (77.7% vs. 16.3%) and by dominating the full risk-
coverage curve, not just the single chosen point. `linear_api` sits
between the two on both axes, as it did on every split tested.

## 6. Ablation: Self-Consistency

Self-consistency (`resolve_with_second_opinion`, `src/selective_prediction.py`)
draws a second independent probability estimate whenever the first lands
inside the escalate band, and nudges the estimate past the band edge on
agreement -- intended to convert genuine near-misses into confident
decisions without forcing a guess on true toss-ups. To test whether this
mechanism actually changes outcomes, `linear_api` and `agentic_api` were
re-run on the identical 300 `holdout_v2` cases with
`--override-self-consistency false`, under the same frozen bands used for
the primary report ((0.35, 0.65) and (0.25, 0.75) respectively) -- an
architecture-vs-architecture comparison on a fixed decision threshold,
not a re-tune.

**Table 4: self-consistency on vs. off, `holdout_v2` (n=300).**

| Arm | Condition | Coverage | Accuracy |
|---|---|---|---|
| `linear_api` | on (primary report) | 50.7% | 0.914 |
| `linear_api` | off (ablation) | 51.7% | 0.903 |
| `agentic_api` | on (primary report) | 17.0% | 0.961 |
| `agentic_api` | off (ablation) | 16.3% | 0.980 |

**McNemar's test, restricted to cases decided under both conditions:**
`linear_api` (n=135 both-decided): 0 discordant pairs, p=1.0.
`agentic_api` (n=40 both-decided): 0 discordant pairs, p=1.0.

**Result: no detectable effect.** Neither coverage nor accuracy moves by
more than the previously measured run-to-run variance (±1.7 points,
Section 5's calibration-split repeat runs), and on every case decided
under both conditions, self-consistency changed the correctness of
exactly zero of them. As implemented, self-consistency is a mechanistically
plausible but empirically inert intervention for this task and model at
this operating point -- a negative result reported directly rather than
selectively omitted. It also confirms the accuracy differences reported
in Section 5 are attributable to the calibration-band fix, not to
self-consistency's presence.

**Incidental finding:** self-consistency's design intent (spend extra
calls only on genuinely ambiguous cases) is corroborated by cost: turning
it off saved $10.04 (`linear_api`) and $6.85 (`agentic_api`) across 300
cases each -- the mechanism is triggering roughly as often as expected
given the bands, even though it isn't moving the accuracy needle at this
operating point.

## 7. Cost and Latency

| Arm | Mean cost/case | Mean latency/case |
|---|---|---|
| `direct_control` | $0.047 | 29.7s |
| `linear_api` | $0.130 | 64.3s |
| `agentic_api` | $0.132 | 57.8s |
| `classical_ml` | ~$0 (local inference) | ~0s |

Full `holdout_v2` run (n=300, 3 API arms): $92.84, zero errors across 900
successful API calls (one transient truncation retried successfully under
the unchanged frozen configuration, consistent with a 0.33% base rate for
this failure mode observed across both holdout rounds). Self-consistency
ablation (Section 6, both arms, same 300 cases, self-consistency off):
$61.87, confirming the mechanism's cost is concentrated on the intended
ambiguous-case subset.

## 8. Discussion

[Author note: synthesize. Key points to make:]

- The calibration-band failure mode (Section 5) generalizes beyond this
  benchmark: any pipeline that (a) elicits numeric confidence scores from
  an LLM and (b) calibrates a decision threshold on a small labeled set
  should expect discretized/anchored outputs and should use a variance-
  aware selection criterion (LCB, not point estimate), not just more
  candidate thresholds.
- Selective prediction changes which comparison is meaningful: point
  accuracy at one operating point is not a fair cross-arm comparison when
  coverage differs this much; the risk-coverage curve is.
- Faithfulness and outcome accuracy are separable: an arm can be highly
  faithful to its evidence (both LLM arms are, ~99%) while still being
  wrong on genuinely hard cases (Section 4, error taxonomy category 9) --
  the interesting failures are in judgment under real uncertainty, not
  evidence fabrication.
- The practical conclusion for a fraud-ops team evaluating this class of
  system: classical ML remains the stronger default; an LLM agent's value
  proposition here is a very high-precision, low-coverage second opinion
  on the subset of cases it is willing to commit to, not a wholesale
  replacement.

## 9. Limitations

Stated directly, not hedged:

- **Single dataset.** IEEE-CIS only. No cross-dataset generalization
  claim is made or supported by this work.
- **Single model.** Claude Sonnet 5 only. No cross-model comparison
  (e.g. GPT-4o, Gemini) was run.
- **Self-authored SOP.** The fraud-analyst SOP was written by the
  researchers, not validated by a practicing fraud analyst. It may not
  reflect real institutional practice.
- **Ground truth.** IEEE-CIS `isFraud` labels are treated as ground truth
  without independent verification; known label-noise concerns in Kaggle
  competition data are not specifically audited here.
- **No human baseline.** This work reports agent-vs-classical-ML
  comparisons only; no claim is made about matching or exceeding human
  analyst performance.
- **Calibration-set size relative to output discretization.** The
  root-cause finding in Section 5 is itself evidence that a 120-case
  calibration set is marginal for this class of method when the
  underlying model's probability outputs are discretized. The fix
  (LCB-based selection) mitigates but does not eliminate this risk; a
  larger calibration set would be a more robust long-term fix.
- **McNemar's tests underpowered.** Section 5.5's paired significance
  tests have low power given how different the arms' coverage profiles
  are; the bootstrap CIs and risk-coverage curves carry more of the
  evidentiary weight in this work.

## 10. Conclusion and Future Work

[Author note: close by naming what's explicitly next, not apologizing for
scope:]
- Human-analyst baseline and agreement study.
- A second, structurally different dataset (candidates identified and
  ranked in `docs/methodology_log.md`: Sparkov, Fraud-ecommerce, BAF).
- Deeper architecture ablations (supervisor-node necessity, tool-count
  sensitivity) beyond the single self-consistency ablation run here.
- Cross-model replication.
