import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_REPORT_PATH = OUTPUTS_DIR / "comparison_report.md"

# Pairwise comparisons worth printing in full, when both sides are present.
KEY_COMPARISONS = [
    ("direct_control", "linear_api"),
    ("linear_api", "agentic_api"),
    ("linear_local", "agentic_local"),
    ("linear_api", "linear_local"),
    ("agentic_api", "agentic_local"),
]

WORKFLOW_COLS = [
    "tools_used_count",
    "checks_completed_count",
    "risk_indicator_count",
    "protective_indicator_count",
    "missing_evidence_count",
]


def discover_arms() -> list[str]:
    return sorted(
        p.stem[: -len("_metrics")]
        for p in OUTPUTS_DIR.glob("*_metrics.csv")
    )


def load_arm(arm: str) -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS_DIR / f"{arm}_metrics.csv")
    df["arm"] = arm
    return df


def summarize_arm(df: pd.DataFrame) -> dict:
    decidable = df[df["predicted_is_fraud"].notna()]
    summary = {
        "n_cases": len(df),
        "n_decidable": len(decidable),
        "mean_latency_ms": df["latency_ms"].mean() if "latency_ms" in df else None,
        "mean_cost_usd": df["cost_usd"].mean() if "cost_usd" in df else None,
        "mean_tool_call_count": df["tool_call_count"].mean() if "tool_call_count" in df else None,
    }

    if len(decidable) > 0:
        y_true = decidable["ground_truth_is_fraud"]
        y_pred = decidable["predicted_is_fraud"].astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        summary["accuracy"] = accuracy_score(y_true, y_pred)
        summary["precision"] = precision
        summary["recall"] = recall
        summary["f1"] = f1
    else:
        summary["accuracy"] = summary["precision"] = summary["recall"] = summary["f1"] = None

    return summary


def _df_to_markdown(df: pd.DataFrame, index_label: str = "") -> str:
    """Hand-rolled markdown table (avoids adding a tabulate dependency
    just for pandas.to_markdown)."""
    if index_label:
        df = df.reset_index()
        if df.columns[0] in ("index", None):
            df = df.rename(columns={df.columns[0]: index_label})

    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for _, row in df.iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=None,
                         help="comma-separated arm names (default: every outputs/*_metrics.csv)")
    parser.add_argument("--save", default=str(DEFAULT_REPORT_PATH),
                         help=f"path to write the markdown report to (default: {DEFAULT_REPORT_PATH})")
    parser.add_argument("--no-save", action="store_true", help="print only, don't write a report file")
    args = parser.parse_args()

    arms = args.arms.split(",") if args.arms else discover_arms()

    if not arms:
        print("No arm metrics found under outputs/*_metrics.csv")
        return

    frames = {}
    for arm in arms:
        try:
            frames[arm] = load_arm(arm)
        except FileNotFoundError:
            print(f"Skipping '{arm}': outputs/{arm}_metrics.csv not found")

    if not frames:
        return

    combined = pd.concat(frames.values(), ignore_index=True)
    report_sections: list[tuple[str, str]] = []

    case_counts = combined.groupby("arm").size().to_frame("n_cases")
    print("\nCase counts:")
    print(case_counts)
    report_sections.append(("Case counts", _df_to_markdown(case_counts, index_label="arm")))

    disposition_counts = pd.crosstab(combined["arm"], combined["disposition"])
    print("\nDisposition counts:")
    print(disposition_counts)
    report_sections.append(("Disposition counts", _df_to_markdown(disposition_counts, index_label="arm")))

    present_workflow_cols = [c for c in WORKFLOW_COLS if c in combined.columns]
    workflow_stats = combined.groupby("arm")[present_workflow_cols].mean()
    print("\nAverage workflow/evidence stats:")
    print(workflow_stats)
    report_sections.append(("Average workflow/evidence stats", _df_to_markdown(workflow_stats, index_label="arm")))

    summary_rows = []
    for arm, df in frames.items():
        summary = summarize_arm(df)
        summary["arm"] = arm
        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows).set_index("arm")[[
        "n_cases", "n_decidable", "accuracy", "precision", "recall", "f1",
        "mean_cost_usd", "mean_latency_ms", "mean_tool_call_count",
    ]]
    print("\nAggregate metrics per arm:")
    print(summary_df)
    report_sections.append(("Aggregate metrics per arm", _df_to_markdown(summary_df, index_label="arm")))

    for arm_a, arm_b in KEY_COMPARISONS:
        if arm_a not in frames or arm_b not in frames:
            continue

        common = frames[arm_a].merge(
            frames[arm_b], on="case_id", suffixes=(f"_{arm_a}", f"_{arm_b}")
        )
        if len(common) == 0:
            continue

        heading = f"{arm_a} vs {arm_b} ({len(common)} common cases)"
        print(f"\n=== {heading} ===")
        gt_col = f"ground_truth_is_fraud_{arm_a}"
        cols = [c for c in [
            "case_id", gt_col, f"disposition_{arm_a}", f"disposition_{arm_b}"
        ] if c in common.columns]
        print(common[cols])
        report_sections.append((heading, _df_to_markdown(common[cols])))

    if args.no_save:
        return

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# FraudOps-Bench comparison report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "**Sample sizes vary by arm below (see Case counts) -- do not read accuracy/cost "
        "differences as conclusive unless the compared arms have comparable n.**",
        "",
    ]
    for heading, table_md in report_sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(table_md)
        lines.append("")

    save_path.write_text("\n".join(lines))
    print(f"\nSaved comparison report to {save_path}")


if __name__ == "__main__":
    main()
