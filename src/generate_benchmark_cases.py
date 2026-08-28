import argparse
import json
from pathlib import Path

import pandas as pd

from train_classical_baseline import load_excluded_transaction_ids, load_merged_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEV_CASES_PATH = PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v0_cases.jsonl"
CALIBRATION_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v1_calibration_cases.jsonl"
HOLDOUT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v1_holdout_cases.jsonl"
HOLDOUT_V2_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v2_holdout_cases.jsonl"
RETRIEVAL_POOL_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fraudops_bench_retrieval_pool_cases.jsonl"

# Same pool criteria the original 50 dev cases were drawn from: card-not-
# present-style rows with some device signal present (device_type/device_info
# feed the visible_case_summary and device_history_check), plus a valid card1
# and transaction amount.
CALIBRATION_N_PER_CLASS = 60   # n=120
HOLDOUT_N_PER_CLASS = 150      # n=300
RETRIEVAL_POOL_N_PER_CLASS = 250  # n=500
CALIBRATION_RANDOM_STATE = 101
HOLDOUT_RANDOM_STATE = 202
HOLDOUT_V2_RANDOM_STATE = 303
RETRIEVAL_POOL_RANDOM_STATE = 404

AVAILABLE_TOOLS = [
    "get_transaction_details",
    "get_card_history",
    "get_email_domain_profile",
    "get_device_history",
    "get_velocity_summary",
    "get_identity_match_summary",
]

REQUIRED_CHECKS = [
    "transaction_amount_check",
    "card_history_check",
    "email_domain_check",
    "device_history_check",
    "velocity_check",
    "identity_consistency_check",
]

EXPECTED_OUTPUT_FIELDS = [
    "disposition",
    "risk_indicators",
    "evidence_used",
    "missing_evidence",
    "final_case_note",
]


def safe_value(x):
    if pd.isna(x):
        return None
    if hasattr(x, "item"):
        return x.item()
    return x


def load_dev_transaction_ids() -> set[int]:
    ids = set()
    with open(DEV_CASES_PATH, "r") as f:
        for line in f:
            ids.add(json.loads(line)["transaction_id"])
    return ids


def build_pool(df: pd.DataFrame, exclude_ids: set[int]) -> pd.DataFrame:
    pool = df[
        (df["DeviceType"].notna() | df["DeviceInfo"].notna())
        & df["card1"].notna()
        & df["TransactionAmt"].notna()
    ]
    return pool[~pool["TransactionID"].isin(exclude_ids)]


def sample_balanced(pool: pd.DataFrame, n_per_class: int, random_state: int) -> pd.DataFrame:
    fraud = pool[pool["isFraud"] == 1].sample(n_per_class, random_state=random_state)
    nonfraud = pool[pool["isFraud"] == 0].sample(n_per_class, random_state=random_state)
    return (
        pd.concat([fraud, nonfraud])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def build_cases(sampled: pd.DataFrame, id_prefix: str) -> list[dict]:
    cases = []
    for i, row in sampled.iterrows():
        cases.append({
            "case_id": f"{id_prefix}_{i + 1:04d}",
            "transaction_id": int(row["TransactionID"]),
            "alert_type": "Card-not-present transaction risk review",
            "ground_truth_is_fraud": int(row["isFraud"]),
            "visible_case_summary": {
                "transaction_amount": safe_value(row["TransactionAmt"]),
                "product_code": safe_value(row["ProductCD"]),
                "transaction_time_delta": safe_value(row["TransactionDT"]),
                "card_brand": safe_value(row["card4"]),
                "card_type": safe_value(row["card6"]),
                "purchaser_email_domain": safe_value(row["P_emaildomain"]),
                "recipient_email_domain": safe_value(row["R_emaildomain"]),
                "billing_region_proxy": safe_value(row["addr1"]),
                "country_proxy": safe_value(row["addr2"]),
                "device_type": safe_value(row["DeviceType"]),
                "device_info": safe_value(row["DeviceInfo"]),
                "os_info": safe_value(row["id_30"]),
                "browser_info": safe_value(row["id_31"]),
                "screen_resolution": safe_value(row["id_33"]),
            },
            "available_tools": AVAILABLE_TOOLS,
            "required_checks": REQUIRED_CHECKS,
            "expected_output_fields": EXPECTED_OUTPUT_FIELDS,
        })
    return cases


def write_jsonl(cases: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")


def generate_calibration_and_holdout():
    print("Loading merged IEEE-CIS dataframe...")
    df = load_merged_dataframe()

    dev_ids = load_dev_transaction_ids()
    print(f"Excluding {len(dev_ids)} existing dev-set transaction_ids from the sampling pool.")

    pool = build_pool(df, exclude_ids=dev_ids)
    print(f"Pool after exclusion: {len(pool)} rows "
          f"({(pool['isFraud'] == 1).sum()} fraud, {(pool['isFraud'] == 0).sum()} non-fraud)")

    calibration_sample = sample_balanced(pool, CALIBRATION_N_PER_CLASS, CALIBRATION_RANDOM_STATE)
    calibration_ids = set(calibration_sample["TransactionID"].astype(int))

    # Holdout is sampled from the pool with calibration's rows removed, so
    # the two new splits are disjoint from each other (and both disjoint
    # from dev, since the pool already excludes dev_ids).
    pool_after_calibration = pool[~pool["TransactionID"].isin(calibration_ids)]
    holdout_sample = sample_balanced(pool_after_calibration, HOLDOUT_N_PER_CLASS, HOLDOUT_RANDOM_STATE)
    holdout_ids = set(holdout_sample["TransactionID"].astype(int))

    assert dev_ids.isdisjoint(calibration_ids), "calibration set overlaps dev set"
    assert dev_ids.isdisjoint(holdout_ids), "holdout set overlaps dev set"
    assert calibration_ids.isdisjoint(holdout_ids), "calibration set overlaps holdout set"

    calibration_cases = build_cases(calibration_sample, "CAL")
    holdout_cases = build_cases(holdout_sample, "HOLD")

    write_jsonl(calibration_cases, CALIBRATION_OUTPUT_PATH)
    write_jsonl(holdout_cases, HOLDOUT_OUTPUT_PATH)

    print(f"Wrote {len(calibration_cases)} calibration cases to {CALIBRATION_OUTPUT_PATH} "
          f"({sum(c['ground_truth_is_fraud'] for c in calibration_cases)} fraud)")
    print(f"Wrote {len(holdout_cases)} holdout cases to {HOLDOUT_OUTPUT_PATH} "
          f"({sum(c['ground_truth_is_fraud'] for c in holdout_cases)} fraud)")


def generate_holdout_v2():
    """A second, fresh holdout set. The original holdout (v1) was scored
    under a calibrate_band() that turned out to overfit small-sample noise
    (see docs/methodology_log.md); its labels were then used to diagnose
    and validate the fix, which spends it as a clean single-use set. This
    generates a disjoint replacement using load_excluded_transaction_ids(),
    which unions dev + calibration + holdout (+ holdout_v2 itself, if a
    prior partial run left a file -- harmless no-op via the same union)."""
    print("Loading merged IEEE-CIS dataframe...")
    df = load_merged_dataframe()

    excluded_ids = load_excluded_transaction_ids()
    print(f"Excluding {len(excluded_ids)} existing benchmark-case transaction_ids "
          f"(dev+calibration+holdout) from the sampling pool.")

    pool = build_pool(df, exclude_ids=excluded_ids)
    print(f"Pool after exclusion: {len(pool)} rows "
          f"({(pool['isFraud'] == 1).sum()} fraud, {(pool['isFraud'] == 0).sum()} non-fraud)")

    holdout_v2_sample = sample_balanced(pool, HOLDOUT_N_PER_CLASS, HOLDOUT_V2_RANDOM_STATE)
    holdout_v2_ids = set(holdout_v2_sample["TransactionID"].astype(int))
    assert excluded_ids.isdisjoint(holdout_v2_ids), "holdout_v2 overlaps an existing split"

    holdout_v2_cases = build_cases(holdout_v2_sample, "HOLD2")
    write_jsonl(holdout_v2_cases, HOLDOUT_V2_OUTPUT_PATH)

    print(f"Wrote {len(holdout_v2_cases)} holdout_v2 cases to {HOLDOUT_V2_OUTPUT_PATH} "
          f"({sum(c['ground_truth_is_fraud'] for c in holdout_v2_cases)} fraud)")


def generate_retrieval_pool():
    """A dedicated, disjoint pool of labeled cases used only as the k-NN
    exemplar source for linear_retrieval/agentic_retrieval (see
    src/retrieval.py). Disjoint from every existing split via
    load_excluded_transaction_ids() (dev+calibration+holdout+holdout_v2),
    same pattern as generate_holdout_v2(). Never itself scored as a case,
    and added to SPLITS in splits.py so it's protected from ever leaking
    into a future split the same way every other split already is."""
    print("Loading merged IEEE-CIS dataframe...")
    df = load_merged_dataframe()

    excluded_ids = load_excluded_transaction_ids()
    print(f"Excluding {len(excluded_ids)} existing benchmark-case transaction_ids "
          f"(dev+calibration+holdout+holdout_v2) from the sampling pool.")

    pool = build_pool(df, exclude_ids=excluded_ids)
    print(f"Pool after exclusion: {len(pool)} rows "
          f"({(pool['isFraud'] == 1).sum()} fraud, {(pool['isFraud'] == 0).sum()} non-fraud)")

    retrieval_pool_sample = sample_balanced(pool, RETRIEVAL_POOL_N_PER_CLASS, RETRIEVAL_POOL_RANDOM_STATE)
    retrieval_pool_ids = set(retrieval_pool_sample["TransactionID"].astype(int))
    assert excluded_ids.isdisjoint(retrieval_pool_ids), "retrieval_pool overlaps an existing split"

    retrieval_pool_cases = build_cases(retrieval_pool_sample, "POOL")
    write_jsonl(retrieval_pool_cases, RETRIEVAL_POOL_OUTPUT_PATH)

    print(f"Wrote {len(retrieval_pool_cases)} retrieval_pool cases to {RETRIEVAL_POOL_OUTPUT_PATH} "
          f"({sum(c['ground_truth_is_fraud'] for c in retrieval_pool_cases)} fraud)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-v2", action="store_true",
                         help="generate the fresh holdout_v2 replacement set instead of calibration+holdout")
    parser.add_argument("--retrieval-pool", action="store_true",
                         help="generate the retrieval_pool exemplar set instead of calibration+holdout")
    args = parser.parse_args()

    if args.holdout_v2:
        generate_holdout_v2()
    elif args.retrieval_pool:
        generate_retrieval_pool()
    else:
        generate_calibration_and_holdout()


if __name__ == "__main__":
    main()
