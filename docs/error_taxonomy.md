# Error taxonomy

Structured categorization of the failure modes found by
`score_faithfulness.py` (deterministic numeric cross-check),
`judge_faithfulness.py` (LLM-judge semantic check, 25-case samples per arm
on `holdout_v2`), and manual review of disposition errors. Source data:
`outputs/holdout_v2/{arm}_faithfulness.csv`,
`outputs/holdout_v2/{arm}_judge_faithfulness.csv`,
`outputs/holdout_v2/{arm}_parsed.jsonl`.

Of the 48 sampled cases (25 `linear_api` + 25 `agentic_api`, 24 scored
each after one skip), the LLM judge flagged 14 individual claims across
14 case-level findings (6/24 `linear_api` cases, 8/24 `agentic_api`
cases). Categorized below.

## 1. Rounding/precision framing (not a real error) -- ~6/14 flagged items

The dominant category, and arguably not an error at all: the analyst
rounds a rate to 1-2 significant figures ("5.3%", "7.8%", "~2.3%") where
the underlying evidence has more precision (5.295%, 7.779%, 2.28%). The
judge consistently flagged these as "misattributions," but this is the
judge being overly literal, not the model misrepresenting evidence -- any
human analyst summarizing a case would round the same way. **Kept as its
own category rather than folded into "real errors" specifically to be
honest about a limitation of the LLM-judge methodology itself**: an
automated judge without an explicit rounding-tolerance instruction will
over-flag benign summarization as fabrication. Worth a prompt fix
(explicit tolerance) if this scoring pass is reused.

> Example (`HOLD2_0129`, `agentic_api`): cited "recipient fraud rate
> 7.78%" against an actual value of 0.07779272238176596 (7.779%).

## 2. Count/quantity miscounting -- 2/14

A genuine, distinct error type: the analyst gets a categorical fact right
in kind but wrong in count.

> `HOLD2_0208` (`agentic_api`): claimed a specific amount "appears twice"
> in the card's prior transaction history; it actually appears three
> times (TransactionIDs 3078948, 3078964, 3078991).

> `HOLD2_0242` (`agentic_api`): claimed the device was "used with 3
> different card identities in rapid succession"; the evidence shows only
> 2 distinct card types.

## 3. Qualitative/comparative mischaracterization -- 2/14

The underlying number is correct, but the descriptive label applied to it
changes its meaning. This is more concerning than a rounding slip, because
it can flip how a human reader weighs the evidence.

> `HOLD2_0112` (`agentic_api`): recipient domain fraud rate (5.15%) --
> actually the *higher* of the two domains being compared -- was
> described as "also low-moderate," implying parity with the purchaser
> domain's 4.35% rather than flagging it as the relatively riskier one.

## 4. Cross-tool / cross-record misattribution -- 2/14

A fact is correctly stated but tied to the wrong source or record.

> `HOLD2_0288` (`linear_api`): cited a specific prior transaction as "one
> prior card transaction" belonging to `card_history`, when that
> transaction ID actually belongs to a different card1 value than the
> current transaction's card.

## 5. Hallucinated field reference -- 1/14

The most concerning single category, though rare: citing a specific,
named structured field that does not exist in the case's actual evidence.

> `HOLD2_0208` (`linear_api`): cited `id_12` as a risk indicator
> ("NotFound"), but `identity_match_summary.identity_flags` for this case
> does not contain an `id_12` entry at all -- only `id_15/16/28/29/35/36/37/38`
> were returned.

## 6. False "consistency/match" claims contradicted by missing data -- 3/14

The analyst asserts two records "match" or are "consistent" when one side
of the comparison is actually null/missing, not equal.

> `HOLD2_0112` (`linear_api`): claimed "the immediately preceding
> transaction on the same card shows a consistent device fingerprint
> (iOS Device, mobile safari)" -- the current transaction's `DeviceInfo`
> is actually null; only the browser string happens to match.

> Same case: claimed the preceding transaction "matches ... same product
> code H" -- the prior transaction's `ProductCD` is actually `'W'`, not
> `'H'`.

## 7. Judge self-contradiction (methodology limitation, not a model error) -- 1/14

`HOLD2_0159` (`agentic_api`): the judge's own listed "why" text concludes
"this is correct, not a misattribution" while still emitting the item in
the `misattributions` list. Left in the raw data rather than silently
dropped -- an honest accounting of LLM-judge noise, not just model noise.

## 8. Derived-arithmetic claims flagged by the numeric pass (not fabrication)

Separate from the judge sample: `score_faithfulness.py`'s deterministic
numeric cross-check flags ~0.7-1.2% of all claimed numbers as "unverified"
across every split run so far (dev, calibration, holdout, holdout_v2).
Manual review (see `docs/methodology_log.md`, 2026-08-16 entry) confirmed
essentially all of these are the model correctly doing arithmetic the
ground-truth flattener doesn't replicate -- e.g. citing "~90 seconds"
between two transactions, computed by subtracting two `TransactionDT`
values that are each individually present in the evidence. Not a
fabrication category; listed here for completeness since it's the other
half of the faithfulness pipeline.

## 9. Evidence-grounded but ultimately incorrect final judgment

Distinct from every category above: cases where every cited fact checks
out (confirmed against `outputs/holdout_v2/agentic_api_parsed.jsonl`) and
the reasoning is coherent, but the final disposition was still wrong. This
is ordinary task difficulty, not a defect in evidence-grounding.

> `HOLD2_0104` (`agentic_api`, false positive): correctly cited a 45%
> confirmed device fraud rate, elevated email-domain fraud rates on both
> sides, an elevated velocity profile, and multiple new/missing identity
> flags -- a genuinely well-corroborated multi-signal risk case by every
> check performed -- and recommended REJECT. Ground truth was non-fraud.
> Every individual claim is accurate; the case was simply a hard one.

## Summary counts

| Category | Count (of 14 judge findings) | Severity |
|---|---|---|
| Rounding/precision framing | ~6 | Not a real error (judge limitation) |
| Count/quantity miscounting | 2 | Real, minor |
| Qualitative/comparative mischaracterization | 2 | Real, moderate |
| Cross-tool/cross-record misattribution | 2 | Real, moderate |
| Hallucinated field reference | 1 | Real, most concerning |
| False consistency/match claims | 3 | Real, moderate-to-concerning |
| Judge self-contradiction | 1 | Judge-side noise |

Net read: once rounding noise and judge self-contradiction are set aside,
both arms show a small number (~8-9 across 48 sampled cases, ~17-19%
case-level rate) of genuine, mostly minor evidence-handling errors --
consistent with the ~99% verified-rate finding from the deterministic
numeric pass. The dominant source of *disposition* errors (category 9) is
ordinary task difficulty on genuinely ambiguous cases, not unfaithful
evidence use.
