import argparse
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

# Import your tools
from tools import (
    get_transaction_details,
    get_card_history,
    get_email_domain_profile,
    get_device_history,
    get_velocity_summary,
    get_identity_match_summary,
)
from splits import cases_path, evidence_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)

    return obj


def build_evidence_packet(case):
    transaction_id = case["transaction_id"]

    return {
        "case_id": case["case_id"],
        "transaction_id": transaction_id,

        # keep ground truth for later evaluation, not for LLM prompt
        "ground_truth_is_fraud": case["ground_truth_is_fraud"],

        "visible_case_summary": case["visible_case_summary"],

        "tool_evidence": {
            "transaction_details": get_transaction_details(transaction_id),
            "card_history": get_card_history(transaction_id),
            "email_domain_profile": get_email_domain_profile(transaction_id),
            "device_history": get_device_history(transaction_id),
            "velocity_summary": get_velocity_summary(transaction_id),
            "identity_match_summary": get_identity_match_summary(transaction_id),
        },

        "required_checks": case["required_checks"],
        "expected_output_fields": case["expected_output_fields"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "calibration", "holdout", "holdout_v2"], default="dev")
    args = parser.parse_args()

    cases = load_jsonl(cases_path(args.split))
    output_path = evidence_path(args.split)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for idx, case in enumerate(cases, start=1):
            print(f"Generating evidence packet {idx}/{len(cases)}: {case['case_id']}")
            packet = build_evidence_packet(case)
            packet = make_json_safe(packet)
            out.write(json.dumps(packet, allow_nan=False) + "\n")

    print(f"Saved evidence packets to {output_path}")


if __name__ == "__main__":
    main()