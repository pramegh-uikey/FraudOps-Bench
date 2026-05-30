import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "outputs" / "evidence_baseline_parsed.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evidence_baseline_metrics.csv"


def map_disposition_to_pred(disposition):
    if disposition == "REJECT":
        return 1
    if disposition == "APPROVE":
        return 0
    if disposition == "ESCALATE":
        return None
    return None


def main():
    rows = []

    with open(INPUT_PATH, "r") as f:
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
            })

    df = pd.DataFrame(rows)

    print("\nParsed successful cases:")
    print(len(df))

    print("\nDisposition counts:")
    print(df["disposition"].value_counts(dropna=False))

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

        print("\nConfusion matrix:")
        print(confusion_matrix(y_true, y_pred))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved case-level metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()