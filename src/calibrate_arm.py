import argparse
import json

from selective_prediction import calibrate_band, save_frozen_band
from splits import arm_output_path

# Reusable, arm-agnostic calibration driver. Previously calibrate_band()/
# save_frozen_band() had no CLI wrapper anywhere in src/ -- the only real
# call sites were notebooks/exp1.ipynb, which is how linear_api/agentic_api/
# classical_ml's bands in configs/calibrated_bands.json were produced. This
# script exists so every arm added from here on (starting with the GPT-5.6
# Terra arms) goes through the same reproducible, non-notebook path.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--target-risk", type=float, default=0.15)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--min-decided-per-band", type=int, default=30)
    args = parser.parse_args()

    input_path = arm_output_path(args.split, args.arm, args.run_tag, suffix="_parsed.jsonl")

    rows = []
    with open(input_path, "r") as f:
        for line in f:
            row = json.loads(line)
            if row.get("parse_status") == "success" and row.get("fraud_probability") is not None:
                rows.append(row)

    if not rows:
        raise SystemExit(
            f"No successfully-parsed rows with a fraud_probability found in {input_path}. "
            "Run run_baseline.py + parse_outputs.py for this arm/split first."
        )

    probabilities = [float(r["fraud_probability"]) for r in rows]
    ground_truth = [int(r["ground_truth_is_fraud"]) for r in rows]

    result = calibrate_band(
        probabilities,
        ground_truth,
        target_risk=args.target_risk,
        k_folds=args.k_folds,
        min_decided_per_band=args.min_decided_per_band,
    )

    selected = result["selected"]
    print(f"Arm: {args.arm} (n={len(rows)} successfully-parsed cases from {input_path})\n")
    print("Full candidate-band sweep (narrowest to widest):")
    for r in sorted(result["sweep"], key=lambda r: r["half_width"]):
        marker = "  <-- selected" if r is selected else ""
        print(
            f"  half_width={r['half_width']:.2f}  band={tuple(round(x, 3) for x in r['band'])}  "
            f"mean_cv_acc={r['mean_cv_accuracy']}  lcb_cv_acc={r['lcb_cv_accuracy']}  "
            f"total_decided={r['total_decided']}{marker}"
        )
    if "warning" in result:
        print(f"\nWARNING: {result['warning']}")

    band = tuple(selected["band"])
    metadata = {
        "calibration_n": len(rows),
        "target_risk": args.target_risk,
        "lcb_cv_accuracy": selected["lcb_cv_accuracy"],
        "mean_cv_accuracy": selected["mean_cv_accuracy"],
        "total_decided": selected["total_decided"],
        "methodology_version": "v2_lcb_fixed_calibrate_band",
    }
    save_frozen_band(args.arm, band, metadata=metadata)
    print(f"\nSelected band {band} saved for arm '{args.arm}' -> configs/calibrated_bands.json")


if __name__ == "__main__":
    main()
