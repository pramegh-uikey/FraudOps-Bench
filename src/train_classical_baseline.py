import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from splits import SPLITS, cases_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
IDENTITY_PATH = PROJECT_ROOT / "data" / "raw" / "train_identity.csv"
CASES_PATH = cases_path("dev")  # kept for backward-compat imports; see load_excluded_transaction_ids for all splits
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classical_ml"

C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_FLAG_COLS = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]  # T/F flags
M_CATEGORICAL_COLS = ["M4"]  # M0/M1/M2, not T/F
ONE_HOT_COLS = ["ProductCD", "card4", "card6", "DeviceType"]

FEATURE_COLUMNS_PATH = OUTPUT_DIR / "feature_columns.json"


def load_excluded_transaction_ids() -> set[int]:
    """Union of transaction_ids across every benchmark split (dev,
    calibration, holdout) -- the classical model must never train on a case
    that any arm gets evaluated on, regardless of which split it's in.
    Splits that haven't been generated yet (e.g. calibration/holdout before
    generate_benchmark_cases.py has run) are skipped rather than erroring,
    so this stays a drop-in replacement before those files exist.
    """
    ids = set()
    for split in SPLITS:
        path = cases_path(split)
        if not path.exists():
            continue
        with open(path, "r") as f:
            for line in f:
                case = json.loads(line)
                ids.add(case["transaction_id"])
    return ids


def load_merged_dataframe() -> pd.DataFrame:
    """Same merge + sort as tools.py, so features here reflect the same
    leakage-free ordering the LLM tool-evidence pipeline uses."""
    df_transaction = pd.read_csv(TRANSACTION_PATH)
    df_identity = pd.read_csv(IDENTITY_PATH)
    df = df_transaction.merge(df_identity, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    return df


def _prior_group_stats(df: pd.DataFrame, key_col: str, prefix: str) -> pd.DataFrame:
    """Leakage-free per-group aggregates: count/fraud-rate/avg-amount of
    rows in the same group that occurred strictly before the current row.

    NOTE: this groups on a single key column (e.g. card1, DeviceInfo,
    P_emaildomain) rather than tools.py's per-row dynamic multi-column card
    mask. That per-row mask can't be vectorized across 590K training rows
    without an expensive per-row loop; grouping on the single strongest
    identifying column is the practical, still-representative approximation
    used for training-scale feature engineering.
    """
    grp = df.groupby(key_col, dropna=False)

    prior_count = grp.cumcount()
    cum_fraud = grp["isFraud"].cumsum() - df["isFraud"]
    cum_amt = grp["TransactionAmt"].cumsum() - df["TransactionAmt"]

    safe_count = prior_count.replace(0, np.nan)

    return pd.DataFrame(
        {
            f"{prefix}_prior_count": prior_count.astype(float),
            f"{prefix}_prior_fraud_rate": (cum_fraud / safe_count),
            f"{prefix}_prior_avg_amt": (cum_amt / safe_count),
        }
    )


def build_features_for_all_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Builds a leakage-free feature matrix for every row in df (already
    merged + sorted by TransactionDT, as returned by load_merged_dataframe).
    Mirrors the evidence surface tools.py exposes to the LLM arms: card/
    device/email-domain fraud-rate history, velocity summary, identity
    flags — not the full raw IEEE-CIS feature dump.
    """
    features = pd.DataFrame(index=df.index)

    features["transaction_amt"] = df["TransactionAmt"]

    card_stats = _prior_group_stats(df, "card1", "card")
    features = pd.concat([features, card_stats], axis=1)
    features["amt_vs_card_avg"] = df["TransactionAmt"] / card_stats["card_prior_avg_amt"]

    device_stats = _prior_group_stats(df, "DeviceInfo", "device")
    features = pd.concat([features, device_stats], axis=1)
    features["amt_vs_device_avg"] = df["TransactionAmt"] / device_stats["device_prior_avg_amt"]

    p_domain_stats = _prior_group_stats(df, "P_emaildomain", "p_domain")
    features = pd.concat([features, p_domain_stats[["p_domain_prior_count", "p_domain_prior_fraud_rate"]]], axis=1)

    r_domain_stats = _prior_group_stats(df, "R_emaildomain", "r_domain")
    features = pd.concat([features, r_domain_stats[["r_domain_prior_count", "r_domain_prior_fraud_rate"]]], axis=1)

    features["domains_match"] = (
        df["P_emaildomain"].notna()
        & df["R_emaildomain"].notna()
        & (df["P_emaildomain"] == df["R_emaildomain"])
    ).astype(float)

    present_c_cols = [c for c in C_COLS if c in df.columns]
    if present_c_cols:
        c_pct_rank = df[present_c_cols].rank(pct=True)
        features["high_c_count_95pct"] = (c_pct_rank >= 0.95).sum(axis=1).astype(float)
        features["max_c_value"] = df[present_c_cols].max(axis=1)
        features["mean_c_value"] = df[present_c_cols].mean(axis=1)

    present_d_cols = [c for c in D_COLS if c in df.columns]
    if present_d_cols:
        features["max_d_value"] = df[present_d_cols].max(axis=1)
        features["mean_d_value"] = df[present_d_cols].mean(axis=1)
        features["missing_d_count"] = df[present_d_cols].isna().sum(axis=1).astype(float)

    present_m_flags = [c for c in M_FLAG_COLS if c in df.columns]
    if present_m_flags:
        m_numeric = df[present_m_flags].map(lambda v: 1.0 if v == "T" else (0.0 if v == "F" else np.nan))
        features["m_flag_true_count"] = (m_numeric == 1.0).sum(axis=1).astype(float)
        features["m_flag_false_count"] = (m_numeric == 0.0).sum(axis=1).astype(float)
        features["m_flag_missing_count"] = m_numeric.isna().sum(axis=1).astype(float)

    one_hot_source = df[[c for c in ONE_HOT_COLS if c in df.columns]]
    if not one_hot_source.empty:
        dummies = pd.get_dummies(one_hot_source, dummy_na=True, prefix=one_hot_source.columns)
        features = pd.concat([features, dummies.astype(float)], axis=1)

    features = features.replace([np.inf, -np.inf], np.nan)

    y = df["isFraud"].astype(int)
    feature_columns = list(features.columns)

    return features, y, feature_columns


def time_based_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
    split_idx = int(len(df) * (1 - test_frac))
    return (
        X.iloc[:split_idx], y.iloc[:split_idx],
        X.iloc[split_idx:], y.iloc[split_idx:],
    )


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name: str):
    if model_name == "logreg":
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ])
    elif model_name == "hgb":
        # HistGradientBoostingClassifier handles NaN natively, no imputation needed.
        pipeline = Pipeline([
            ("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=0)),
        ])
    else:
        raise ValueError(f"Unknown model_name '{model_name}', expected 'logreg' or 'hgb'")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n[{model_name}] holdout accuracy={acc:.3f} precision={precision:.3f} "
          f"recall={recall:.3f} f1={f1:.3f} auc={auc:.3f}")

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "hgb", "both"], default="both")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading merged dataframe...")
    df = load_merged_dataframe()

    excluded_ids = load_excluded_transaction_ids()
    before = len(df)
    df = df[~df["TransactionID"].isin(excluded_ids)].reset_index(drop=True)
    print(f"Excluded {before - len(df)} benchmark-case transactions (dev+calibration+holdout) "
          f"from training (expected up to {len(excluded_ids)}).")
    assert df["TransactionID"].isin(excluded_ids).sum() == 0, "benchmark transaction IDs leaked into training data"

    print("Building leakage-free features for all transactions...")
    X, y, feature_columns = build_features_for_all_transactions(df)

    X_train, y_train, X_test, y_test = time_based_split(df, X, y, test_frac=args.test_frac)
    print(f"Train rows: {len(X_train)}, holdout rows: {len(X_test)}, "
          f"train fraud rate: {y_train.mean():.4f}, holdout fraud rate: {y_test.mean():.4f}")

    models_to_train = ["logreg", "hgb"] if args.model == "both" else [args.model]

    for model_name in models_to_train:
        pipeline = train_and_evaluate(X_train, y_train, X_test, y_test, model_name)
        model_path = out_dir / f"{model_name}_model.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved {model_name} model to {model_path}")

    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f)
    print(f"Saved feature column list to {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
