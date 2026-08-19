import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from selective_prediction import load_frozen_band, probability_to_disposition, risk_coverage
from splits import arm_output_path, outputs_dir

RNG_SEED = 0
N_BOOT = 10000

ARM_BANDS = {
    "linear_api": (0.35, 0.65),
    "agentic_api": (0.25, 0.75),
    "classical_ml": (0.35, 0.65),
}


def load_probs(split: str, arm: str, run_tag: str | None = None) -> pd.DataFrame:
    path = arm_output_path(split, arm, run_tag, suffix="_parsed.jsonl")
    rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame([r for r in rows if r.get("parse_status") == "success"])
    band = ARM_BANDS.get(arm) or load_frozen_band(arm) or (0.4, 0.6)
    df["disposition"] = df["fraud_probability"].apply(lambda p: probability_to_disposition(float(p), band))
    df["predicted_is_fraud"] = df["disposition"].map({"REJECT": 1, "APPROVE": 0})
    return df


def mcnemar_test(y_true: pd.Series, pred_a: pd.Series, pred_b: pd.Series) -> dict:
    """McNemar's exact test on paired correctness: is arm A significantly
    more/less accurate than arm B on the same cases? b = A right, B wrong;
    c = A wrong, B right. Restricted by the caller to cases both arms
    actually decided (excludes ESCALATE) -- accuracy isn't defined for an
    undecided case."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(((correct_a) & (~correct_b)).sum())
    c = int(((~correct_a) & (correct_b)).sum())
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    result = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": result.pvalue}


def bootstrap_ci(df: pd.DataFrame, n_boot: int = N_BOOT, ci: float = 0.95, seed: int = RNG_SEED) -> dict:
    """Percentile bootstrap over the FULL case set (including escalated
    cases), resampling case rows with replacement each draw so both
    coverage and accuracy-on-decided vary together, the way they would if
    a genuinely different sample of cases had been drawn."""
    rng = np.random.default_rng(seed)
    n = len(df)
    acc_samples, cov_samples, covw_samples = [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        decided = sample[sample["predicted_is_fraud"].notna()]
        coverage = len(decided) / n
        if len(decided) > 0:
            accuracy = (decided["predicted_is_fraud"] == decided["ground_truth_is_fraud"]).mean()
        else:
            accuracy = np.nan
        acc_samples.append(accuracy)
        cov_samples.append(coverage)
        covw_samples.append(coverage * accuracy if not np.isnan(accuracy) else np.nan)

    def _pct_ci(values):
        values = np.array([v for v in values if not np.isnan(v)])
        lo, hi = np.percentile(values, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
        return float(np.mean(values)), float(lo), float(hi)

    acc_mean, acc_lo, acc_hi = _pct_ci(acc_samples)
    cov_mean, cov_lo, cov_hi = _pct_ci(cov_samples)
    covw_mean, covw_lo, covw_hi = _pct_ci(covw_samples)
    return {
        "accuracy": (acc_mean, acc_lo, acc_hi),
        "coverage": (cov_mean, cov_lo, cov_hi),
        "coverage_weighted": (covw_mean, covw_lo, covw_hi),
    }


def risk_coverage_curve(df: pd.DataFrame, half_widths=None) -> list[dict]:
    if half_widths is None:
        half_widths = np.arange(0.0, 0.51, 0.025)
    probs = df["fraud_probability"].astype(float).to_numpy()
    gt = df["ground_truth_is_fraud"].to_numpy()
    curve = []
    for h in half_widths:
        band = (0.5 - h, 0.5 + h)
        rc = risk_coverage(probs, gt, band)
        curve.append({"half_width": float(h), "coverage": rc["coverage"], "accuracy": rc["accuracy"]})
    return curve


def aurc(curve: list[dict]) -> float:
    """Area under the risk-coverage curve (risk = 1 - accuracy), sorted by
    coverage and integrated via the trapezoid rule. Lower is better."""
    points = [(c["coverage"], 1 - c["accuracy"]) for c in curve if c["accuracy"] is not None]
    points.sort(key=lambda p: p[0])
    if len(points) < 2:
        return float("nan")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(np.trapezoid(ys, xs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="holdout_v2")
    args = parser.parse_args()

    arms = ["linear_api", "agentic_api", "classical_ml"]
    data = {arm: load_probs(args.split, arm) for arm in arms}

    report_lines = [f"# Statistical analysis -- {args.split}\n"]

    # --- Bootstrap CIs ---
    report_lines.append("## Bootstrap 95% CIs (10,000 resamples, case-level)\n")
    report_lines.append("| Arm | Accuracy | Coverage | Coverage-weighted |")
    report_lines.append("|---|---|---|---|")
    for arm in arms:
        ci = bootstrap_ci(data[arm])
        acc = ci["accuracy"]
        cov = ci["coverage"]
        covw = ci["coverage_weighted"]
        report_lines.append(
            f"| {arm} | {acc[0]:.3f} [{acc[1]:.3f}, {acc[2]:.3f}] "
            f"| {cov[0]:.3f} [{cov[1]:.3f}, {cov[2]:.3f}] "
            f"| {covw[0]:.3f} [{covw[1]:.3f}, {covw[2]:.3f}] |"
        )
    report_lines.append("")

    # --- Pairwise McNemar's tests (on the intersection of cases both arms decided) ---
    report_lines.append("## Pairwise McNemar's tests (paired, cases both arms decided)\n")
    report_lines.append("| Comparison | n (both decided) | b | c | p-value |")
    report_lines.append("|---|---|---|---|---|")
    pairs = [("linear_api", "agentic_api"), ("linear_api", "classical_ml"), ("agentic_api", "classical_ml")]
    for arm_a, arm_b in pairs:
        merged = data[arm_a][["case_id", "ground_truth_is_fraud", "predicted_is_fraud"]].merge(
            data[arm_b][["case_id", "predicted_is_fraud"]], on="case_id", suffixes=(f"_{arm_a}", f"_{arm_b}")
        )
        both_decided = merged.dropna(subset=[f"predicted_is_fraud_{arm_a}", f"predicted_is_fraud_{arm_b}"])
        result = mcnemar_test(
            both_decided["ground_truth_is_fraud"],
            both_decided[f"predicted_is_fraud_{arm_a}"],
            both_decided[f"predicted_is_fraud_{arm_b}"],
        )
        report_lines.append(
            f"| {arm_a} vs {arm_b} | {len(both_decided)} | {result['b']} | {result['c']} | {result['p_value']:.4f} |"
        )
    report_lines.append(
        "\n*Note: arms with very different coverage (e.g. `agentic_api` at 17% vs "
        "`classical_ml` at 85%) share few commonly-decided cases, so these paired "
        "tests are underpowered by construction -- a near-1.0 p-value here reflects "
        "small-n, not evidence of equivalence. The bootstrap CIs and risk-coverage "
        "curves above are the more informative comparison when coverage differs "
        "this much.*"
    )
    report_lines.append("")

    # --- Risk-coverage curves + AURC ---
    report_lines.append("## Risk-coverage curves (descriptive; actual reported operating point marked)\n")
    report_lines.append("| Arm | AURC (lower=better) | Operating-point half-width | Operating-point coverage | Operating-point accuracy |")
    report_lines.append("|---|---|---|---|---|")

    plt.figure(figsize=(7, 5))
    for arm in arms:
        curve = risk_coverage_curve(data[arm])
        area = aurc(curve)
        low, high = ARM_BANDS[arm]
        op_h = round((high - low) / 2, 3)
        op_point = next((c for c in curve if abs(c["half_width"] - op_h) < 1e-6), None)
        covs = [c["coverage"] for c in curve if c["accuracy"] is not None]
        accs = [c["accuracy"] for c in curve if c["accuracy"] is not None]
        plt.plot(covs, accs, marker="o", markersize=3, label=arm)
        if op_point:
            plt.scatter([op_point["coverage"]], [op_point["accuracy"]], s=120, edgecolor="black", zorder=5)
            report_lines.append(
                f"| {arm} | {area:.4f} | {op_h} | {op_point['coverage']:.3f} | {op_point['accuracy']:.3f} |"
            )

    plt.xlabel("Coverage (fraction of cases decided)")
    plt.ylabel("Accuracy on decided cases")
    plt.title(f"Risk-coverage curves -- {args.split}")
    plt.legend()
    plt.grid(alpha=0.3)
    fig_path = outputs_dir(args.split) / "risk_coverage_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    report_lines.append(f"\nFigure saved to `{fig_path}`\n")

    report_path = outputs_dir(args.split) / "stats_report.md"
    report_path.write_text("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
