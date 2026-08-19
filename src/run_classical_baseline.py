import argparse
import json
from pathlib import Path

import joblib

from selective_prediction import probability_to_disposition
from splits import arm_output_path, cases_path
from train_classical_baseline import (
    OUTPUT_DIR,
    build_features_for_all_transactions,
    load_merged_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "calibration", "holdout", "holdout_v2"], default="dev")
    parser.add_argument("--model-path", default=str(OUTPUT_DIR / "hgb_model.joblib"))
    parser.add_argument("--escalate-band", default="0.4,0.6",
                         help="comma-separated low,high probability band mapped to ESCALATE")
    args = parser.parse_args()

    low, high = (float(x) for x in args.escalate_band.split(","))

    output_path = arm_output_path(args.split, "classical_ml", suffix="_parsed.jsonl")

    pipeline = joblib.load(args.model_path)
    model_name = Path(args.model_path).stem

    print("Loading merged dataframe and rebuilding features for scoring...")
    df = load_merged_dataframe()
    X, y, _ = build_features_for_all_transactions(df)

    cases = load_jsonl(cases_path(args.split))
    case_by_txn_id = {c["transaction_id"]: c for c in cases}
    target_ids = set(case_by_txn_id.keys())

    mask = df["TransactionID"].isin(target_ids)
    target_df = df[mask]
    target_X = X[mask]
    probabilities = pipeline.predict_proba(target_X)[:, 1]

    rows = []
    for txn_id, proba in zip(target_df["TransactionID"], probabilities):
        case = case_by_txn_id.get(txn_id)
        if case is None:
            continue

        disposition = probability_to_disposition(proba, (low, high))

        rows.append({
            "case_id": case["case_id"],
            "transaction_id": txn_id,
            "ground_truth_is_fraud": case["ground_truth_is_fraud"],
            "arm": "classical_ml",
            "flow": "classical_ml",
            "model": model_name,
            "parse_status": "success",
            "fraud_probability": float(proba),
            "disposition": disposition,
            "risk_indicators": [],
            "protective_indicators": [],
            "tools_used": [],
            "required_checks_completed": [],
            "missing_evidence": [],
            "evidence_used": [],
            "final_case_note": f"predicted_fraud_probability={proba:.4f}",
            "latency_ms": 0.0,
            "input_tokens": None,
            "output_tokens": None,
            "cost_usd": 0.0,
            "tool_call_count": None,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")

    print(f"Scored {len(rows)}/{len(cases)} '{args.split}' cases, saved to {output_path}")


if __name__ == "__main__":
    main()
