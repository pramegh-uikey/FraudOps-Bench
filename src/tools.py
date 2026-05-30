import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
IDENTITY_PATH = PROJECT_ROOT / "data" / "raw" / "train_identity.csv"

df_transaction = pd.read_csv(TRANSACTION_PATH)
df_identity = pd.read_csv(IDENTITY_PATH)

df = df_transaction.merge(df_identity, on="TransactionID", how="left")
df = df.sort_values("TransactionDT").reset_index(drop=True)


def safe_json_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x

def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return None
    return float(numerator / denominator)

def get_transaction_details(transaction_id):
    row = df[df["TransactionID"] == transaction_id]

    if row.empty:
        return {"error": "TransactionID not found"}

    row = row.iloc[0]

    fields = [
        "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "dist2",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"
    ]

    return {col: safe_json_value(row[col]) for col in fields if col in df.columns}


def get_card_history(transaction_id, max_history=20):
    row = df[df["TransactionID"] == transaction_id]

    if row.empty:
        return {"error": "TransactionID not found"}

    row = row.iloc[0]
    card1 = row["card1"]
    current_time = row["TransactionDT"]

    card_cols = ["card1", "card2", "card3", "card4", "card5", "card6"]
    card_cols = [col for col in card_cols if col in df.columns and pd.notna(row[col])]

    mask = pd.Series(True, index=df.index)

    for col in card_cols:
        mask = mask & (df[col] == row[col])

    hist = df[
        mask &
        (df["TransactionDT"] < current_time)
    ].sort_values("TransactionDT").tail(max_history)

    recent_cols = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_30",
        "id_31",
        "id_33"
    ]

    recent_cols = [col for col in recent_cols if col in hist.columns]

    return {
        "card_proxy": safe_json_value(card1),
        "prior_transaction_count": int(len(hist)),
        "card_proxy_fields_used": card_cols,

        # Analyst-facing aggregate history
        "confirmed_prior_fraud_count": int(hist["isFraud"].sum()) if len(hist) else 0,
        "confirmed_prior_fraud_rate": float(hist["isFraud"].mean()) if len(hist) else None,

        "avg_prior_amount": float(hist["TransactionAmt"].mean()) if len(hist) else None,
        "max_prior_amount": float(hist["TransactionAmt"].max()) if len(hist) else None,
        "current_amount_vs_avg_prior_amount": (
            safe_ratio(row["TransactionAmt"], hist["TransactionAmt"].mean())
            if len(hist)
            else None
        ),

        # No raw isFraud exposed here
        "recent_transactions": hist[recent_cols].map(safe_json_value).to_dict("records")
    }


def get_email_domain_profile(transaction_id):
    row = df[df["TransactionID"] == transaction_id]

    if row.empty:
        return {"error": "TransactionID not found"}

    row = row.iloc[0]
    p_domain = row["P_emaildomain"]
    r_domain = row["R_emaildomain"]

    p_hist = df[df["P_emaildomain"] == p_domain] if pd.notna(p_domain) else pd.DataFrame()
    r_hist = df[df["R_emaildomain"] == r_domain] if pd.notna(r_domain) else pd.DataFrame()

    return {
        "purchaser_email_domain": safe_json_value(p_domain),
        "recipient_email_domain": safe_json_value(r_domain),
        "domains_match": bool(p_domain == r_domain) if pd.notna(p_domain) and pd.notna(r_domain) else None,
        "p_domain_transaction_count": int(len(p_hist)),
        "p_domain_fraud_rate": float(p_hist["isFraud"].mean()) if len(p_hist) else None,
        "r_domain_transaction_count": int(len(r_hist)),
        "r_domain_fraud_rate": float(r_hist["isFraud"].mean()) if len(r_hist) else None
    }


def get_device_history(transaction_id, max_history=20):
    row = df[df["TransactionID"] == transaction_id]

    if row.empty:
        return {"error": "TransactionID not found"}

    row = row.iloc[0]
    device_info = row["DeviceInfo"]
    device_type = row["DeviceType"]
    current_time = row["TransactionDT"]

    if pd.isna(device_info):
        return {
            "device_info": None,
            "device_type": safe_json_value(device_type),
            "message": "No DeviceInfo available for this transaction"
        }

    device_cols = ["DeviceInfo", "DeviceType", "id_30", "id_31", "id_33"]
    device_cols = [col for col in device_cols if col in df.columns and pd.notna(row[col])]

    mask = pd.Series(True, index=df.index)

    for col in device_cols:
        mask = mask & (df[col] == row[col])

    hist = df[
        mask &
        (df["TransactionDT"] < current_time)
    ].sort_values("TransactionDT").tail(max_history)

    recent_cols = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_30",
        "id_31",
        "id_33"
    ]

    recent_cols = [col for col in recent_cols if col in hist.columns]

    return {
        "device_info": safe_json_value(device_info),
        "device_type": safe_json_value(device_type),
        "device_proxy_fields_used": device_cols,
        "prior_device_transaction_count": int(len(hist)),

        # Analyst-facing aggregate history
        "confirmed_prior_device_fraud_count": int(hist["isFraud"].sum()) if len(hist) else 0,
        "confirmed_prior_device_fraud_rate": float(hist["isFraud"].mean()) if len(hist) else None,

        "avg_prior_device_amount": float(hist["TransactionAmt"].mean()) if len(hist) else None,
        "max_prior_device_amount": float(hist["TransactionAmt"].max()) if len(hist) else None,
        "current_amount_vs_avg_prior_device_amount": (
            safe_ratio(row["TransactionAmt"], hist["TransactionAmt"].mean())
            if len(hist)
            else None
        ),

        # No raw isFraud exposed here
        "recent_device_transactions": hist[recent_cols].map(safe_json_value).to_dict("records")
    }


def get_velocity_summary(transaction_id):
    row = df[df["TransactionID"] == transaction_id]

    if row.empty:
        return {"error": "TransactionID not found"}

    row = row.iloc[0]

    c_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in df.columns]
    d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in df.columns]

    c_percentiles = {}
    d_missing_count = int(row[d_cols].isna().sum())

    for col in c_cols:
        val = row[col]
        if pd.isna(val):
            c_percentiles[col] = None
        else:
            c_percentiles[col] = float((df[col] <= val).mean())

    high_c_count = sum(
        1 for v in c_percentiles.values()
        if v is not None and v >= 0.95
    )

    return {
        "count_feature_summary": {
            "high_count_feature_count_95th_pct": high_c_count,
            "max_C_value": safe_json_value(row[c_cols].max()),
            "mean_C_value": safe_json_value(row[c_cols].mean())
        },
        "time_delta_feature_summary": {
            "max_D_value": safe_json_value(row[d_cols].max()),
            "mean_D_value": safe_json_value(row[d_cols].mean()),
            "missing_D_count": d_missing_count
        }
    }

def get_identity_match_summary(transaction_id):
    row = df[df["TransactionID"] == transaction_id]

    if row.empty:
        return {"error": "TransactionID not found"}

    row = row.iloc[0]

    m_cols = [f"M{i}" for i in range(1, 10) if f"M{i}" in df.columns]
    id_flag_cols = ["id_12", "id_15", "id_16", "id_28", "id_29", "id_35", "id_36", "id_37", "id_38"]
    id_flag_cols = [col for col in id_flag_cols if col in df.columns]

    return {
        "match_features": {col: safe_json_value(row[col]) for col in m_cols},
        "identity_flags": {col: safe_json_value(row[col]) for col in id_flag_cols},
        "device_type": safe_json_value(row.get("DeviceType")),
        "device_info": safe_json_value(row.get("DeviceInfo")),
        "os_info": safe_json_value(row.get("id_30")),
        "browser_info": safe_json_value(row.get("id_31")),
        "screen_resolution": safe_json_value(row.get("id_33"))
    }