import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def map_disposition_to_pred(disposition):
    if disposition == "REJECT":
        return 1
    if disposition == "APPROVE":
        return 0
    if disposition == "ESCALATE":
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / "outputs" / f"{args.arm}_parsed.jsonl"
    output_path = PROJECT_ROOT / "outputs" / f"{args.arm}_metrics.csv"

    rows = []

    with open(input_path, "r") as f:
        for line in f:
            row = json.loads(line)

            if row.get("parse_status") != "success":
                continue

            disposition = row.get("disposition")
            pred = map_disposition_to_pred(disposition)

            rows.append({
                "case_id": row["case_id"],
                "ground_truth_is_fraud": row["ground_truth_is_fraud"],
                "disposition": disposition,
                "predicted_is_fraud": pred,
                "tools_used_count": len(row.get("tools_used", [])),
                "checks_completed_count": len(row.get("required_checks_completed", [])),
                "risk_indicator_count": len(row.get("risk_indicators", [])),
                "protective_indicator_count": len(row.get("protective_indicators", [])),
                "missing_evidence_count": len(row.get("missing_evidence", [])),
                "latency_ms": row.get("latency_ms"),
                "cost_usd": row.get("cost_usd"),
                "tool_call_count": row.get("tool_call_count"),
            })

    df = pd.DataFrame(rows)

    print(f"\nArm: {args.arm}")

    print("\nParsed successful cases:")
    print(len(df))

    if len(df) == 0:
        print("No successful parsed cases.")
        return

    print("\nDisposition counts:")
    print(df["disposition"].value_counts(dropna=False))

    print("\nMean cost/latency/tool-call-count:")
    print(df[["latency_ms", "cost_usd", "tool_call_count"]].mean(numeric_only=True))

    decidable = df[df["predicted_is_fraud"].notna()].copy()

    print("\nDecidable cases only:")
    print(len(decidable))

    if len(decidable) > 0:
        y_true = decidable["ground_truth_is_fraud"]
        y_pred = decidable["predicted_is_fraud"].astype(int)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        print("\nOutcome metrics on APPROVE/REJECT only:")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1:        {f1:.3f}")

        print("\nConfusion matrix labels=[0,1]:")
        print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    df.to_csv(output_path, index=False)
    print(f"\nSaved case-level metrics to {output_path}")


if __name__ == "__main__":
    main()
