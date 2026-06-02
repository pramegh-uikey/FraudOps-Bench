import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIRECT_PATH = PROJECT_ROOT / "outputs" / "direct_baseline_metrics.csv"
EVIDENCE_PATH = PROJECT_ROOT / "outputs" / "evidence_baseline_metrics.csv"

direct = pd.read_csv(DIRECT_PATH)
evidence = pd.read_csv(EVIDENCE_PATH)

direct["baseline"] = "direct"
evidence["baseline"] = "evidence"

df = pd.concat([direct, evidence], ignore_index=True)

print("\nCase counts:")
print(df.groupby("baseline").size())

print("\nDisposition counts:")
print(pd.crosstab(df["baseline"], df["disposition"]))

print("\nAverage workflow/evidence stats:")
cols = [
    "tools_used_count",
    "checks_completed_count",
    "risk_indicator_count",
    "protective_indicator_count",
    "missing_evidence_count",
]
print(df.groupby("baseline")[cols].mean())

common = direct.merge(
    evidence,
    on="case_id",
    suffixes=("_direct", "_evidence")
)

print("\nCommon cases:")
print(len(common))

if len(common):
    print("\nDisposition changes:")
    print(
        common[
            [
                "case_id",
                "ground_truth_is_fraud_direct",
                "disposition_direct",
                "disposition_evidence",
            ]
        ]
    )