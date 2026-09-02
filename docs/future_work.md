# Future work

Deliberately deferred ideas for after the IEEE Access submission
(`paper/latex/fraudops_bench.tex`) is submitted and accepted -- explicit
user instruction, 2026-08-20 (reaffirmed 2026-08-28). Do not start item
3 (still open) before then. All of them target the same root cause:
Section 5.3's finding that `agentic_api`'s `fraud_probability` outputs
are discretized (cluster on round, prompt-anchored values) rather than
continuous, which is what breaks small-sample threshold calibration in
the first place. Items 1 and 2 were both run as submission-timeline
exceptions (racing the deadline, `main` kept as the untouched fallback)
-- see each item for its branch and outcome.

## 1. Retrieval-augmented exemplars -- DONE (2026-08-29), result mixed

Built and run on the `retrieval-exemplars` branch (not merged into
`main` as of this writing -- a submission-timeline call, see the branch
and `paper/latex/fraudops_bench.tex` Section 5.8 for the full writeup).
This item is no longer open; recorded here for the record and because
its actual result directly motivates item 3 below.

**What was tried:** a dedicated `retrieval_pool` split (n=500, disjoint
from every existing split, free/local to build), k=5 nearest neighbors
via Euclidean distance over a standardized feature vector from each
case's own evidence surface, injected as a "similar past cases" block
into both the linear and agentic prompts.

**What actually happened, not what was hypothesized:** the motivating
idea -- that grounding in retrieved exemplars would produce a less
discretized, more continuous `fraud_probability` -- did not hold on
inspection. `agentic_retrieval`'s raw probability distribution is
nearly identical to `agentic_api`'s (same dominant round values, ~50%
concentration in the top 3 values either way). `linear_retrieval`
showed a real but modest reduction. Despite that, both arms showed a
genuine practical improvement: `agentic_retrieval` reached the best
full-curve LLM result in the paper (AURC 0.0896 vs.\ `agentic_api`'s
0.1117), nearly tripling coverage (45.0% vs.\ 17.0%) for a 2.8-point
accuracy cost. `linear_retrieval` improved on AURC too but not on
coverage-weighted correctness at its own operating point -- a genuinely
mixed result. Both arms paid a real cost in LLM-judge-flagged semantic
faithfulness (56.0% and 41.7% of sampled cases showed a misattribution,
vs.\ 33.3% and 25.0% without retrieval).

**The open question this leaves:** if it isn't smoother probabilities
producing the gain, the more plausible account is that retrieval
grounding improves judgment *within* the same discretized value
buckets, not the granularity of the buckets themselves -- unconfirmed.
There is also no repeat-run stochasticity check for these two arms
(unlike Section 5.1a's Claude estimate), so some part of the reported
gain could in principle be run-to-run variance rather than a stable
effect. See `paper/latex/fraudops_bench.tex` Section 5.8 and Section 9
(Limitations) for the full, honest treatment.

**Prior art it built on:** FinFRE-RAG (Tan, Ma, and Zhang, 2025 --
already cited as `\cite{finfrerag2025}` in Related Work) does retrieval-
augmented in-context learning for LLM fraud *classification* (F1/MCC
only, no agent, no selective prediction). This imported their retrieval
approach into FraudOps-Bench's agentic/selective-prediction framework,
which neither paper had tested.

## 2. Sampling-based continuous probability -- DONE (2026-09-02), result genuinely messy

Built and run on the `vote-probability-check` branch (not merged into
`main`, same submission-timeline call as item 1). See
`docs/methodology_log.md`'s 2026-09-02 entry for the full numbers and
narrative; summarized here.

**What was tried:** stop asking the model to *verbalize* a single
`fraud_probability` number and instead sample it N=10 times on a simple
binary question ("fraud or not?"), using the vote fraction as the
probability -- naturally continuous at 1/N granularity, no verbalization
step to anchor on a round number. Run on `linear_api`'s single-shot
flow, `dev` split (n=50), paired against a freshly-run verbalized
baseline on the identical 50 cases.

**What actually happened:** the opposite of the hypothesis, and not
subtly. Plain vote-fraction (no reasoning step before answering)
collapsed to 4 unique values across 50 cases (98.0% top-3 concentration,
96% landing at exactly 0.0 or 1.0) versus the verbalized baseline's 24
unique values (32.7% top-3, 0% at the extremes). A confound was
identified and tested directly: the vote prompt had dropped the
verbalized prompt's six-required-check reasoning step. A control run
restoring that reasoning (changing only the final answer format) showed
a real, substantial effect -- 9 unique values, 74.0% top-3 -- confirming
the missing reasoning mattered. But even controlling for it, voting
remained far more bimodal than verbalized probability (64% still at the
exact extremes), and manual inspection of raw responses revealed the
measurement itself is messier than assumed: the model mostly ignores
"answer one word only" and reverts to its full trained JSON response,
appending a final word that is sometimes inconsistent with its own
internal `fraud_probability`/`disposition` (one case: internal 0.5/
ESCALATE, appended word a flat "FRAUD"). Scaffolded-vote accuracy (62%)
came in below both the verbalized baseline (71%) and even the plain
vote (76%).

**What this means for item 3 (RL):** not encouraging, and not a clean
answer either way. Voting does not turn out to be a cheap fix for
discretization -- if anything it's a different, more extreme
discretization failure mode (near-total bimodal certainty rather than
graded uncertainty). Worse, getting the model to reliably separate
"structured reasoning" from "constrained final answer format" is itself
harder than the experiment design assumed, which is a genuine
complication for any RL reward built on "the model can express
fine-grained confidence if only asked correctly" -- that premise looks
shakier after this result, not more solid. Total cost: ~$69 across both
runs plus verification spot-checks.

## 3. Reinforcement learning on the calibration signal

**Sequencing note, updated now that item 2 is done:** the calibration
reward design below rests on the premise that discretization is a
fixable verbalization artifact, addressable by rewarding genuinely
differentiated probabilities. Item 2's result is a real caution against
that premise, not confirmation of it -- the model's repeated independent
judgments collapsed toward binary certainty rather than showing the
graded uncertainty a Brier-score reward would need to shape, and the
model didn't even reliably separate reasoning from final-answer format
when asked to. This doesn't rule RL out, but it means going in without
illusions: this may be fighting the model's actual judgment structure,
not a superficial habit. If pursued, start with a small-scale pilot
specifically testing whether the reward can move the needle at all
before committing to a full training run.

**Idea:** instead of prompting a frozen pretrained model, let it learn
from a reward signal computed against real fraud/not-fraud labels.

**Two candidate reward designs (can be combined):**
- **Calibration reward.** Score each stated `fraud_probability` against
  the true label with a proper scoring rule (e.g. Brier score / log
  loss) -- this directly punishes overconfident, round-number-anchored
  guesses and rewards genuinely differentiated probabilities. Attacks
  Section 5.3's discretization at the source, rather than working around
  it with retrieval or the LCB fix.
- **End-to-end decision reward.** Fold in the selective-prediction rule
  itself: correct APPROVE/REJECT scores well, a confidently wrong call
  scores badly, ESCALATE is a small, safe middle score.

**Two paths, very different cost:**
1. **Hosted reinforcement fine-tuning (lower effort).** OpenAI offers a
   reinforcement fine-tuning service where you supply the reward formula
   and cases and they run the RL loop -- no training infrastructure to
   manage. GPT-5.6 Terra is already wired into the pipeline
   (`llm_backends.py`), making this the more approachable entry point.
2. **Open-weight model, full control (higher effort).** Train
   `linear_local`/`agentic_local` (built but never run -- see paper
   Limitations) with real RL training infrastructure (e.g. GRPO/PPO) and
   GPU compute. Fully inspectable and cheaper per-call afterward, but a
   real infrastructure lift.

**Honest scope check:** RL needs many more labeled training cases than
the current 120-case calibration set to learn anything nontrivial.
IEEE-CIS has ~590k transactions, so raw data isn't the constraint --
building enough properly-formatted, SOP-consistent training cases (tool
evidence, check verdicts) at that scale is the real work. This is a
third-paper-scale project, not a one-week add-on.
