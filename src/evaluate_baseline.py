import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from selective_prediction import DEFAULT_BAND, load_frozen_band, probability_to_disposition
from splits import arm_output_path

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
    parser.add_argument("--split", choices=["dev", "calibration", "holdout", "holdout_v2"], default="dev")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--band", default=None,
                         help="comma-separated low,high escalate band applied to fraud_probability "
                              "(recomputed at eval time, independent of the band used at generation "
                              "time -- lets you sweep bands without regenerating LLM calls). "
                              "Default: the frozen calibration-split band for this arm if one exists, "
                              "else DEFAULT_BAND (with a warning).")
    args = parser.parse_args()

    if args.band is not None:
        band = tuple(float(x) for x in args.band.split(","))
    else:
        frozen = load_frozen_band(args.arm)
        if frozen is not None:
            band = frozen
            print(f"Using frozen calibration-split band for '{args.arm}': {band}")
        else:
            band = DEFAULT_BAND
            print(f"No frozen band found for '{args.arm}' -- falling back to DEFAULT_BAND={DEFAULT_BAND}. "
                  f"Run band calibration on the calibration split and save_frozen_band() first "
                  f"for a non-default, non-ad-hoc band.")

    input_path = arm_output_path(args.split, args.arm, args.run_tag, suffix="_parsed.jsonl")
    output_path = arm_output_path(args.split, args.arm, args.run_tag, suffix="_metrics.csv")

    rows = []

    with open(input_path, "r") as f:
        for line in f:
            row = json.loads(line)

            if row.get("parse_status") != "success":
                continue

            fraud_probability = row.get("fraud_probability")
            if fraud_probability is not None:
                # Calibrated path: disposition is a deterministic function of
                # the probability + band, not the model's own free-form pick.
                disposition = probability_to_disposition(float(fraud_probability), band)
            else:
                # Legacy fallback: no probability field parsed (e.g. older
                # data, or a row where fraud_probability failed to parse) --
                # trust the model's own categorical disposition.
                disposition = row.get("disposition")

            pred = map_disposition_to_pred(disposition)

            rows.append({
                "case_id": row["case_id"],
                "ground_truth_is_fraud": row["ground_truth_is_fraud"],
                "disposition": disposition,
                "disposition_raw": row.get("disposition_raw", row.get("disposition")),
                "fraud_probability": fraud_probability,
                "self_consistency_triggered": bool(row.get("self_consistency_check", {}).get("triggered")),
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

    print(f"\nArm: {args.arm} (band={band})")

    print("\nParsed successful cases:")
    print(len(df))

    if len(df) == 0:
        print("No successful parsed cases.")
        return

    print("\nDisposition counts (calibrated):")
    print(df["disposition"].value_counts(dropna=False))

    if "disposition_raw" in df.columns:
        print("\nDisposition counts (model's own raw call, for comparison):")
        print(df["disposition_raw"].value_counts(dropna=False))

    if df["self_consistency_triggered"].any():
        print(f"\nSelf-consistency second opinion triggered on "
              f"{int(df['self_consistency_triggered'].sum())}/{len(df)} cases")

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
