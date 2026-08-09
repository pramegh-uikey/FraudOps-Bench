import argparse
import json
from pathlib import Path

import joblib

from train_classical_baseline import (
    CASES_PATH,
    OUTPUT_DIR,
    build_features_for_all_transactions,
    load_merged_dataframe,
    load_pilot_transaction_ids,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "classical_ml_parsed.jsonl"


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def probability_to_disposition(p: float, escalate_band: tuple[float, float]) -> str:
    low, high = escalate_band
    if p >= high:
        return "REJECT"
    if p <= low:
        return "APPROVE"
    return "ESCALATE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(OUTPUT_DIR / "hgb_model.joblib"))
    parser.add_argument("--escalate-band", default="0.4,0.6",
                         help="comma-separated low,high probability band mapped to ESCALATE")
    args = parser.parse_args()

    low, high = (float(x) for x in args.escalate_band.split(","))

    pipeline = joblib.load(args.model_path)
    model_name = Path(args.model_path).stem

    print("Loading merged dataframe and rebuilding features for scoring...")
    df = load_merged_dataframe()
    X, y, _ = build_features_for_all_transactions(df)

    pilot_ids = load_pilot_transaction_ids()
    cases = load_jsonl(CASES_PATH)
    case_by_txn_id = {c["transaction_id"]: c for c in cases}

    mask = df["TransactionID"].isin(pilot_ids)
    pilot_df = df[mask]
    pilot_X = X[mask]
    probabilities = pipeline.predict_proba(pilot_X)[:, 1]

    rows = []
    for txn_id, proba in zip(pilot_df["TransactionID"], probabilities):
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")

    print(f"Scored {len(rows)}/{len(cases)} pilot cases, saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
