# Future work

Deliberately deferred ideas for after the IEEE Access submission
(`paper/latex/fraudops_bench.tex`) is submitted and accepted -- explicit
user instruction, 2026-08-20 (reaffirmed 2026-08-28). Do not start any of
the still-open items below before then. All of them target the same
root cause: Section 5.3's finding that `agentic_api`'s
`fraud_probability` outputs are discretized (cluster on round,
prompt-anchored values) rather than continuous, which is what breaks
small-sample threshold calibration in the first place.

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

## 2. Sampling-based continuous probability (bypass verbalization entirely)

**Idea, motivated directly by item 1's actual result:** stop asking the
model to *verbalize* a single `fraud_probability` number at all --
that's the step that anchors it to round, prompt-shaped values in the
first place. Instead, sample the model N times (e.g. N=10) at
temperature > 0 on a simpler binary question ("fraud or not?"), and use
the vote fraction as the probability estimate. A fraction like 7/10 is
naturally continuous at 1/N granularity and never passes through a
verbalization step that could anchor on a round number.

**Why this is the right next experiment, not RL:** item 1 showed that
grounding the model in better context (retrieval) improves outcomes
*without* fixing discretization -- so the discretization problem itself
is still open, and it's still unclear whether it's fixable at all short
of retraining. This test answers that directly and cheaply: no
training, no reward-hacking risk, just more inference calls per case.
If vote-fraction probabilities show measurably less round-number
clustering than verbalized ones, that's real evidence the problem is a
verbalization artifact (and that a Brier-score RL reward, item 3 below,
is targeting a sound mechanism). If they don't, that's evidence the
clustering runs deeper than how confidence gets phrased, and item 3's
premise needs rethinking before spending real time/money on it.

**Cost/effort:** cheap. No new arm-calibration cycle needed to get a
first read -- run N-sample voting on a modest sample of already-scored
holdout cases and compare the resulting probability histogram's
round-number concentration against the existing verbalized-probability
baselines already on file (`outputs/holdout_v2/agentic_api_parsed.jsonl`
etc.). A day or two, not a week.

## 3. Reinforcement learning on the calibration signal

**Sequencing: do item 2 first.** This reward design's whole premise --
that discretization is a fixable verbalization artifact -- is exactly
what item 2 is designed to test cheaply, without training anything. Item
1's actual result already showed one intervention (retrieval) improving
outcomes without fixing discretization; don't commit RL budget to a
Brier-score reward until item 2 gives a reason to believe fixing
discretization is possible at all.

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
