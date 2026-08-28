import argparse
import json

import pandas as pd

from faithfulness import (
    load_agentic_evidence_by_case,
    load_jsonl,
    load_linear_evidence_by_case,
    score_case,
)
from splits import arm_output_path

SUPPORTED_ARMS = ["linear_api", "agentic_api", "linear_gpt", "agentic_gpt", "linear_retrieval", "agentic_retrieval"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=SUPPORTED_ARMS)
    parser.add_argument("--split", choices=["dev", "calibration", "holdout", "holdout_v2"], default="dev")
    args = parser.parse_args()

    parsed_path = arm_output_path(args.split, args.arm, suffix="_parsed.jsonl")
    parsed_rows = load_jsonl(parsed_path)

    if args.arm in ("linear_api", "linear_gpt", "linear_retrieval"):
        evidence_by_case = load_linear_evidence_by_case(args.split)
        tools_called_by_case = {}
    else:
        agentic_data = load_agentic_evidence_by_case(args.split, args.arm)
        evidence_by_case = {cid: ev for cid, (ev, _called) in agentic_data.items()}
        tools_called_by_case = {cid: called for cid, (_ev, called) in agentic_data.items()}

    rows = []
    skipped = 0
    for row in parsed_rows:
        if row.get("parse_status") != "success":
            skipped += 1
            continue
        case_id = row["case_id"]
        ground_truth = evidence_by_case.get(case_id)
        if not ground_truth:
            skipped += 1
            continue
        tools_called = tools_called_by_case.get(case_id)
        score = score_case(row, ground_truth, tools_called)
        score["case_id"] = case_id
        rows.append(score)

    output_path = arm_output_path(args.split, args.arm, suffix="_faithfulness.csv")
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Arm: {args.arm} (split={args.split})")
    print(f"Scored {len(rows)} cases, skipped {skipped} (parse failures or missing evidence)")
    if len(rows) == 0:
        print("No cases scored.")
        return

    scored_with_claims = df[df["n_claimed_numbers"] > 0]
    print(f"\nCases with >=1 extractable numeric claim: {len(scored_with_claims)}/{len(df)}")
    if len(scored_with_claims) > 0:
        print(f"Mean verified_rate (among those cases): {scored_with_claims['verified_rate'].mean():.3f}")
    print(f"Total claimed numbers: {int(df['n_claimed_numbers'].sum())}, "
          f"verified: {int(df['n_verified'].sum())}, "
          f"unverified (candidate fabrications): {int(df['n_unverified'].sum())}")
    n_any_unverified = (df["n_unverified"] > 0).sum()
    print(f"Cases with >=1 unverified numeric claim: {n_any_unverified}/{len(df)}")

    print(f"\nDotted-style citations found: {int(df['n_dotted_citations'].sum())}, "
          f"exact-match correct: {int(df['n_dotted_correct'].sum())}")
    print(f"Uninformative citations (tool named, no number): {int(df['n_uninformative_citations'].sum())}")
    if args.arm in ("agentic_api", "agentic_gpt"):
        print(f"Citations referencing a tool never actually called: {int(df['n_uncalled_tool_citations'].sum())}")

    print(f"\nSaved case-level faithfulness scores to {output_path}")


if __name__ == "__main__":
    main()
