# Future work

Deliberately deferred ideas for after the IEEE Access submission
(`paper/latex/fraudops_bench.tex`) is submitted and accepted -- explicit
user instruction, 2026-08-20 (reaffirmed 2026-08-28). Do not start any of
this before then. Both ideas below target the same root cause: Section
5.3's finding that `agentic_api`'s `fraud_probability` outputs are
discretized (cluster on round, prompt-anchored values) rather than
continuous, which is what breaks small-sample threshold calibration in
the first place.

## 1. Retrieval-augmented exemplars

**Idea:** before the agent states a `fraud_probability`, retrieve a
handful (~5) of similar past cases from the dev/calibration pool --
matched on features like amount, device pattern, velocity signature --
along with their known ground-truth outcomes, and give the agent that
comparison as extra evidence (e.g. "4 of 5 similar past cases were
fraud").

**Why it's supposed to help:** a verbalized probability is essentially
the model guessing a number in words, which is what produces the
round-number anchoring diagnosed in Section 5.3. A nearest-neighbor
count ("4/5 similar cases were fraud" -> 0.8) is a naturally finer-
grained, less discretized signal. If retrieval genuinely smooths that
clustering out, the LCB-based calibration fix (Section 5.4) gets a
better-behaved probability distribution to work with in the first
place, rather than just adding a safety margin around a discretized one.

**Why it could plausibly beat `classical_ml`, not just tie it:**
gradient boosting already does something like nearest-neighbor reasoning
internally (splits on feature similarity), so retrieval alone isn't a
fundamentally new signal. The honest case for winning is combining that
similarity signal with cross-evidence narrative reasoning a GBT's
additive splits can't do (e.g. "this matches known fraud patterns on
velocity, but the identity-match signal here is unusually strong, which
those matched cases didn't have"). This is a real hypothesis to test,
not a guaranteed win -- flagged as a genuine engineering lift with
uncertain payoff.

**What it would take:** build a case-retrieval index over the
dev/calibration pool, redo calibration with retrieval-augmented
probability estimates, check whether clustering at round values
(0.45/0.55-style edges) is reduced or just moved elsewhere.

**Prior art to build on:** FinFRE-RAG (Tan, Ma, and Zhang, 2025 --
already cited as `\cite{finfrerag2025}` in Related Work) does retrieval-
augmented in-context learning for LLM fraud *classification* (F1/MCC
only, no agent, no selective prediction). This idea imports their
retrieval approach into FraudOps-Bench's agentic/selective-prediction
framework to attack the calibration failure mode directly, which neither
paper currently tests.

## 2. Reinforcement learning on the calibration signal

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
third-paper-scale project, likely after or alongside the retrieval-
exemplar idea above, not a one-week add-on.
