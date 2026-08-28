import argparse
import json
import random

from faithfulness import (
    TOOL_NAME_TO_EVIDENCE_KEY,
    load_agentic_evidence_by_case,
    load_jsonl,
    load_linear_evidence_by_case,
)
from llm_backends import call_llm
from parsing import extract_json
from splits import arm_output_path

JUDGE_MODEL = "claude-haiku-4-5"

# The numeric cross-referencing pass (score_faithfulness.py) catches
# fabricated numbers, but not semantic misattribution: a claim that cites a
# real number correctly while asserting something false about what it means
# (e.g. "domains match" when they don't, or crediting device_history with a
# fact that actually came from card_history -- exactly what the SOP's
# "Evidence attribution rules" section warns against). This is cheap enough
# (~$0.01-0.05/case on Haiku) to run as a small spot-check sample rather
# than every case.
JUDGE_PROMPT = """You are auditing a fraud analyst's case notes against the actual tool evidence they had access to. Your job is ONLY to check factual/attribution accuracy -- not to second-guess the analyst's final judgment call.

REAL TOOL EVIDENCE (ground truth):
{ground_truth_json}

ANALYST'S CITED EVIDENCE AND REASONING:
{claims_text}

For each distinct factual claim in the analyst's reasoning above, check it against the real tool evidence. Flag a claim as "misattributed" if it:
- States something that contradicts the real evidence (e.g. says domains match when they don't, says a rate is high when it's actually low)
- Correctly cites a real number but attributes it to the wrong source/field/meaning
- Draws a conclusion the evidence doesn't actually support

Do NOT flag: reasonable inferences, derived arithmetic (e.g. computing a time gap from two timestamps), or claims you cannot verify one way or the other from the evidence given (mark those "unverifiable", not "misattributed").

Return ONLY valid JSON, no markdown fences:
{{
  "misattributions": [
    {{"claim": "short quote or paraphrase of the claim", "why": "one sentence explaining the contradiction"}}
  ],
  "unverifiable_count": 0,
  "total_claims_reviewed": 0
}}

If there are no misattributions, return an empty list for "misattributions"."""


def _claims_text(model_output: dict) -> str:
    parts = []
    for field in ["evidence_used", "risk_indicators", "protective_indicators"]:
        for item in (model_output.get(field) or []):
            parts.append(f"- [{field}] {item}")
    for cv in (model_output.get("check_verdicts") or []):
        if isinstance(cv, dict) and cv.get("detail"):
            parts.append(f"- [check_verdicts:{cv.get('check')}] ({cv.get('verdict')}) {cv['detail']}")
    if model_output.get("final_case_note"):
        parts.append(f"- [final_case_note] {model_output['final_case_note']}")
    return "\n".join(parts)


def judge_case(model_output: dict, ground_truth_evidence: dict) -> dict:
    # Show the judge the FULL evidence, including recent_transactions /
    # recent_device_transactions -- the model's own prompt includes these
    # (see tools.py's get_card_history/get_device_history), and it
    # legitimately cites specific past-transaction details from them. An
    # earlier version of this function hard-truncated the serialized JSON
    # at 8000 characters (silently dropping later tools) and a version
    # after that stripped these history arrays entirely to control size --
    # both caused the judge to flag real, verifiable claims as fabricated
    # simply because it couldn't see the evidence that supported them.
    # Haiku's context window comfortably fits the full packet (a few
    # thousand tokens at most), so there's no real reason to trim it.
    prompt = JUDGE_PROMPT.format(
        ground_truth_json=json.dumps(ground_truth_evidence, indent=2, default=str),
        claims_text=_claims_text(model_output),
    )
    result = call_llm("anthropic", JUDGE_MODEL, prompt, max_tokens=1024)
    if result.error is not None or result.raw_text is None:
        return {"error": result.error, "cost_usd": result.cost_usd}
    try:
        parsed = extract_json(result.raw_text)
    except Exception as e:
        return {"error": f"judge response parse failure: {e}", "cost_usd": result.cost_usd}
    parsed["cost_usd"] = result.cost_usd
    parsed["error"] = None
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True,
                         choices=["linear_api", "agentic_api", "linear_gpt", "agentic_gpt",
                                  "linear_retrieval", "agentic_retrieval"])
    parser.add_argument("--split", required=True,
                         choices=["dev", "calibration", "holdout", "holdout_v2"])
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    parsed_rows = load_jsonl(arm_output_path(args.split, args.arm, suffix="_parsed.jsonl"))
    parsed_rows = [r for r in parsed_rows if r.get("parse_status") == "success"]

    if args.arm in ("linear_api", "linear_gpt", "linear_retrieval"):
        evidence_by_case = load_linear_evidence_by_case(args.split)
    else:
        agentic_data = load_agentic_evidence_by_case(args.split, args.arm)
        evidence_by_case = {cid: ev for cid, (ev, _called) in agentic_data.items()}

    random.seed(args.seed)
    sample = random.sample(parsed_rows, min(args.sample_size, len(parsed_rows)))

    output_path = arm_output_path(args.split, args.arm, suffix="_judge_faithfulness.csv")
    total_cost = 0.0
    rows = []

    for row in sample:
        case_id = row["case_id"]
        ground_truth = evidence_by_case.get(case_id)
        if not ground_truth:
            continue
        result = judge_case(row, ground_truth)
        cost = result.get("cost_usd") or 0.0
        total_cost += cost
        misattributions = result.get("misattributions", [])
        rows.append({
            "case_id": case_id,
            "n_misattributions": len(misattributions),
            "misattributions": json.dumps(misattributions),
            "unverifiable_count": result.get("unverifiable_count"),
            "total_claims_reviewed": result.get("total_claims_reviewed"),
            "error": result.get("error"),
            "cost_usd": cost,
        })
        print(f"{case_id}: {len(misattributions)} misattribution(s), "
              f"{result.get('unverifiable_count')} unverifiable, cost=${cost:.4f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_scored = len(df[df["error"].isna()]) if "error" in df.columns else len(df)
    n_with_misattribution = int((df["n_misattributions"] > 0).sum()) if len(df) else 0
    print(f"\nJudged {n_scored}/{len(sample)} sampled cases for '{args.arm}' ({args.split})")
    print(f"Cases with >=1 misattribution: {n_with_misattribution}/{n_scored}")
    print(f"Total judge cost: ${total_cost:.2f}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
