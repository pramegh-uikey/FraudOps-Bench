import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import load_models_config
from splits import outputs_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def _known_arm_names() -> list[str]:
    return sorted(load_models_config()["arms"].keys())


def group_metrics_files_by_arm(split: str) -> dict[str, list[Path]]:
    """Maps each known arm name to every {arm}[_{run_tag}]_metrics.csv file
    present for this split, so repeat runs (different --run-tag) are
    aggregated instead of treated as unrelated arms. Matching is against
    the arm names in configs/models.yaml since arm names themselves can
    contain underscores, making pure filename-splitting ambiguous."""
    files = sorted(outputs_dir(split).glob("*_metrics.csv"))
    stems = {p: p.stem[: -len("_metrics")] for p in files}

    grouped: dict[str, list[Path]] = {}
    for path, stem in stems.items():
        for arm in _known_arm_names():
            if stem == arm or stem.startswith(f"{arm}_"):
                grouped.setdefault(arm, []).append(path)
                break
        else:
            # Not a recognized arm name (e.g. config changed since the run) --
            # still surface it standalone under its raw stem.
            grouped.setdefault(stem, []).append(path)

    return grouped


def discover_arms(split: str = "dev") -> list[str]:
    return sorted(group_metrics_files_by_arm(split).keys())


def load_arm(split: str, arm: str, files_by_arm: dict[str, list[Path]] | None = None) -> pd.DataFrame:
    """Pooled dataframe across every repeat-run file for this arm (single
    file when there are no repeats)."""
    if files_by_arm is None:
        files_by_arm = group_metrics_files_by_arm(split)
    frames = [pd.read_csv(p) for p in files_by_arm[arm]]
    df = pd.concat(frames, ignore_index=True)
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
    parser.add_argument("--split", choices=["dev", "calibration", "holdout", "holdout_v2"], default="dev")
    parser.add_argument("--arms", default=None,
                         help="comma-separated arm names (default: every outputs/{split}/*_metrics.csv)")
    parser.add_argument("--save", default=None,
                         help="path to write the markdown report to (default: outputs/{split}/comparison_report.md)")
    parser.add_argument("--no-save", action="store_true", help="print only, don't write a report file")
    args = parser.parse_args()

    split_outputs_dir = outputs_dir(args.split)
    default_report_path = split_outputs_dir / "comparison_report.md"

    files_by_arm = group_metrics_files_by_arm(args.split)
    arms = args.arms.split(",") if args.arms else sorted(files_by_arm.keys())

    if not arms:
        print(f"No arm metrics found under {split_outputs_dir}/*_metrics.csv")
        return

    frames = {}
    for arm in arms:
        if arm not in files_by_arm:
            print(f"Skipping '{arm}': no {split_outputs_dir}/{arm}*_metrics.csv found")
            continue
        frames[arm] = load_arm(args.split, arm, files_by_arm)

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

    faithfulness_rows = []
    for arm in frames:
        fpath = split_outputs_dir / f"{arm}_faithfulness.csv"
        if not fpath.exists():
            continue
        fdf = pd.read_csv(fpath)
        with_claims = fdf[fdf["n_claimed_numbers"] > 0]
        faithfulness_rows.append({
            "arm": arm,
            "n_cases_scored": len(fdf),
            "mean_verified_rate": with_claims["verified_rate"].mean() if len(with_claims) else None,
            "cases_with_unverified_claim": int((fdf["n_unverified"] > 0).sum()),
            "uninformative_citations": int(fdf["n_uninformative_citations"].sum()),
        })
    if faithfulness_rows:
        faithfulness_df = pd.DataFrame(faithfulness_rows).set_index("arm")
        print("\nFaithfulness (evidence-attribution scoring, where available):")
        print(faithfulness_df)
        report_sections.append(("Faithfulness (evidence-attribution scoring)",
                                 _df_to_markdown(faithfulness_df, index_label="arm")))

    repeat_arms = {arm: paths for arm, paths in files_by_arm.items() if arm in frames and len(paths) > 1}
    if repeat_arms:
        variance_rows = []
        for arm, paths in repeat_arms.items():
            per_run = [summarize_arm(pd.read_csv(p)) for p in paths]
            row = {"arm": arm, "n_runs": len(per_run)}
            for metric in ["accuracy", "precision", "recall", "f1"]:
                values = [r[metric] for r in per_run if r[metric] is not None]
                row[f"{metric}_mean"] = float(np.mean(values)) if values else None
                row[f"{metric}_std"] = float(np.std(values)) if len(values) > 1 else None
            variance_rows.append(row)
        variance_df = pd.DataFrame(variance_rows).set_index("arm")
        print(f"\nRepeat-run variance ({len(repeat_arms)} arm(s) with >1 run):")
        print(variance_df)
        report_sections.append(("Repeat-run variance across --run-tag repeats", _df_to_markdown(variance_df, index_label="arm")))

    for arm_a, arm_b in KEY_COMPARISONS:
        if arm_a not in frames or arm_b not in frames:
            continue
        if arm_a in repeat_arms or arm_b in repeat_arms:
            # Per-case merge isn't meaningful across pooled repeat runs
            # (case_id repeats once per run) -- skip, variance section above
            # already covers these arms.
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

    save_path = Path(args.save) if args.save else default_report_path
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
