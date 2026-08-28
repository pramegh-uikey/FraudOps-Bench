"""k-NN retrieval of similar past cases from the frozen retrieval_pool
split, used by linear_retrieval/agentic_retrieval to ground a case's
fraud_probability in comparable prior outcomes instead of an unaided
verbalized guess (see docs/future_work.md and paper Section 5.3's
discretization diagnosis, which this arm is testing a fix for).

Feature basis is deliberately independent of classical_ml's
build_features_for_all_transactions() (train_classical_baseline.py):
that function is batch-only cumulative-stats over the full transaction
history and has real leakage-sensitivity baked into its ordering
assumptions. Retrieval here instead uses only fields already present in
a case's own evidence packet (the same surface the LLM/analyst sees),
which keeps the two arms' comparison clean and avoids coupling to
classical_ml's internals.

K and the feature set are frozen here before any calibration run --
tuning either against calibration accuracy later would reproduce the
exact small-sample overfitting mistake Section 5.3 documents, one level
up. Covered by freeze_methodology.py's METHODOLOGY_FILES.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from splits import cases_path, evidence_path
from tools import (
    get_card_history,
    get_device_history,
    get_email_domain_profile,
    get_identity_match_summary,
    get_velocity_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

K_NEIGHBORS = 5

NUMERIC_FIELDS = [
    ("visible_case_summary", "transaction_amount"),
    ("tool_evidence.card_history", "confirmed_prior_fraud_rate"),
    ("tool_evidence.card_history", "avg_prior_amount"),
    ("tool_evidence.card_history", "current_amount_vs_avg_prior_amount"),
    ("tool_evidence.card_history", "prior_transaction_count"),
    ("tool_evidence.email_domain_profile", "p_domain_fraud_rate"),
    ("tool_evidence.email_domain_profile", "r_domain_fraud_rate"),
    ("tool_evidence.device_history", "confirmed_prior_device_fraud_rate"),
    ("tool_evidence.device_history", "avg_prior_device_amount"),
    ("tool_evidence.device_history", "prior_device_transaction_count"),
    ("tool_evidence.velocity_summary.count_feature_summary", "high_count_feature_count_95th_pct"),
    ("tool_evidence.velocity_summary.count_feature_summary", "max_C_value"),
    ("tool_evidence.velocity_summary.count_feature_summary", "mean_C_value"),
    ("tool_evidence.velocity_summary.time_delta_feature_summary", "max_D_value"),
    ("tool_evidence.velocity_summary.time_delta_feature_summary", "mean_D_value"),
    ("tool_evidence.velocity_summary.time_delta_feature_summary", "missing_D_count"),
]

CATEGORICAL_FIELDS = [
    ("visible_case_summary", "product_code"),
    ("visible_case_summary", "card_brand"),
    ("visible_case_summary", "card_type"),
    ("visible_case_summary", "device_type"),
]

# M1-M9 identity-match flags: True/False/missing -> per-case summary
# counts, mirroring classical_ml's m_flag_true/false/missing_count
# treatment (train_classical_baseline.py) at a case-summary level rather
# than per-column, keeping the feature vector small and stable even if a
# given M-column is sometimes absent from tool_evidence.


def _get_path(obj, dotted_path):
    node = obj
    for part in dotted_path.split("."):
        if node is None:
            return None
        node = node.get(part)
    return node


def _get_field(packet, spec):
    group, field = spec
    node = _get_path(packet, group)
    if node is None:
        return None
    return node.get(field)


def _m_flag_counts(packet):
    match_features = _get_path(packet, "tool_evidence.identity_match_summary.match_features") or {}
    true_count = sum(1 for v in match_features.values() if v == "T" or v is True)
    false_count = sum(1 for v in match_features.values() if v == "F" or v is False)
    missing_count = sum(1 for v in match_features.values() if v is None)
    return true_count, false_count, missing_count


def _domains_match_value(packet):
    val = _get_path(packet, "tool_evidence.email_domain_profile.domains_match")
    if val is True:
        return 1.0
    if val is False:
        return 0.0
    return 0.5  # unknown/None -- neutral, not imputed away


def build_query_packet(case: dict) -> dict:
    """For agentic_retrieval: the agent gathers its own tool evidence
    on-graph, so unlike linear_retrieval it has no pre-built evidence
    packet to query with. This computes the same tool_evidence shape
    directly (same six deterministic, no-LLM-cost functions
    agentic_graph.py's tool nodes call), purely for the retrieval
    feature vector -- it is never shown to the agent itself, which still
    must call its own tools for its own reasoning."""
    transaction_id = case["transaction_id"]
    return {
        "transaction_id": transaction_id,
        "visible_case_summary": case["visible_case_summary"],
        "tool_evidence": {
            "card_history": get_card_history(transaction_id),
            "email_domain_profile": get_email_domain_profile(transaction_id),
            "device_history": get_device_history(transaction_id),
            "velocity_summary": get_velocity_summary(transaction_id),
            "identity_match_summary": get_identity_match_summary(transaction_id),
        },
    }


def load_pool_packets() -> list[dict]:
    """Loads the frozen retrieval_pool's evidence packets (must have been
    generated via generate_evidence_packets.py --split retrieval_pool)."""
    path = evidence_path("retrieval_pool")
    packets = []
    with open(path, "r") as f:
        for line in f:
            packets.append(json.loads(line))
    return packets


def _extract_numeric_row(packet) -> list[float]:
    row = [_get_field(packet, spec) for spec in NUMERIC_FIELDS]
    row.append(_domains_match_value(packet))
    t, fl, m = _m_flag_counts(packet)
    row.extend([t, fl, m])
    return [np.nan if v is None else float(v) for v in row]


class RetrievalIndex:
    """Fits once on retrieval_pool, then answers nearest-neighbor queries
    for arbitrary new cases (calibration/holdout) via .query(packet)."""

    def __init__(self):
        self.pool_packets = load_pool_packets()
        if len(self.pool_packets) < K_NEIGHBORS:
            raise ValueError(
                f"retrieval_pool has only {len(self.pool_packets)} cases, "
                f"need at least {K_NEIGHBORS}"
            )
        self.pool_ids = {p["transaction_id"] for p in self.pool_packets}

        numeric_matrix = np.array([_extract_numeric_row(p) for p in self.pool_packets])
        self.imputer = SimpleImputer(strategy="mean")
        numeric_imputed = self.imputer.fit_transform(numeric_matrix)
        self.scaler = StandardScaler()
        numeric_scaled = self.scaler.fit_transform(numeric_imputed)

        self.categories = {}
        cat_columns = []
        for spec in CATEGORICAL_FIELDS:
            values = [_get_field(p, spec) or "__missing__" for p in self.pool_packets]
            cats = sorted(set(values))
            self.categories[spec] = cats
            one_hot = np.zeros((len(self.pool_packets), len(cats)))
            for i, v in enumerate(values):
                one_hot[i, cats.index(v)] = 1.0
            cat_columns.append(one_hot)

        self.feature_matrix = np.hstack([numeric_scaled] + cat_columns)
        self.nn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="euclidean")
        self.nn.fit(self.feature_matrix)

    def _vectorize_one(self, packet) -> np.ndarray:
        numeric_row = np.array([_extract_numeric_row(packet)])
        numeric_imputed = self.imputer.transform(numeric_row)
        numeric_scaled = self.scaler.transform(numeric_imputed)

        cat_columns = []
        for spec in CATEGORICAL_FIELDS:
            cats = self.categories[spec]
            value = _get_field(packet, spec) or "__missing__"
            one_hot = np.zeros((1, len(cats)))
            if value in cats:
                one_hot[0, cats.index(value)] = 1.0
            # unseen category at query time -> all-zero row, a neutral
            # "unlike anything in the pool on this dimension" signal
            cat_columns.append(one_hot)

        return np.hstack([numeric_scaled] + cat_columns)

    def query(self, packet, k: int = K_NEIGHBORS) -> list[dict]:
        if packet["transaction_id"] in self.pool_ids:
            raise ValueError(
                f"transaction_id {packet['transaction_id']} is itself in "
                f"retrieval_pool -- refusing to query (would leak the case's "
                f"own label back to itself)."
            )
        vec = self._vectorize_one(packet)
        distances, indices = self.nn.kneighbors(vec, n_neighbors=k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            pool_packet = self.pool_packets[idx]
            results.append({
                "case_id": pool_packet["case_id"],
                "transaction_id": pool_packet["transaction_id"],
                "ground_truth_is_fraud": pool_packet["ground_truth_is_fraud"],
                "distance": float(dist),
                "transaction_amount": _get_path(pool_packet, "visible_case_summary.transaction_amount"),
                "product_code": _get_path(pool_packet, "visible_case_summary.product_code"),
                "device_type": _get_path(pool_packet, "visible_case_summary.device_type"),
            })
        return results


def format_exemplars_for_prompt(neighbors: list[dict]) -> str:
    n_fraud = sum(1 for n in neighbors if n["ground_truth_is_fraud"] == 1)
    n_total = len(neighbors)
    lines = [
        f"SIMILAR PAST CASES (retrieved by nearest-neighbor similarity on "
        f"transaction amount, card/device/email history, and velocity "
        f"signals; {n_fraud} of {n_total} were confirmed fraud):"
    ]
    for i, n in enumerate(neighbors, start=1):
        outcome = "FRAUD" if n["ground_truth_is_fraud"] == 1 else "NOT FRAUD"
        lines.append(
            f"{i}. amount=${n['transaction_amount']}, product={n['product_code']}, "
            f"device={n['device_type']} -> confirmed outcome: {outcome}"
        )
    return "\n".join(lines)


_INDEX_SINGLETON = None


def get_index() -> RetrievalIndex:
    global _INDEX_SINGLETON
    if _INDEX_SINGLETON is None:
        _INDEX_SINGLETON = RetrievalIndex()
    return _INDEX_SINGLETON


def get_exemplar_text(packet, k: int = K_NEIGHBORS) -> str:
    index = get_index()
    neighbors = index.query(packet, k=k)
    return format_exemplars_for_prompt(neighbors)


def get_exemplar_text_for_case(case: dict, k: int = K_NEIGHBORS) -> str:
    """agentic_retrieval's entry point -- builds the query packet on the
    fly from the raw case (see build_query_packet) rather than requiring
    a pre-generated evidence packet."""
    packet = build_query_packet(case)
    return get_exemplar_text(packet, k=k)


if __name__ == "__main__":
    # Smoke test: build the index and query it for the first few
    # calibration-split cases (never part of the pool, safe to test with).
    idx = get_index()
    print(f"Fitted retrieval index on {len(idx.pool_packets)} pool cases, "
          f"feature dim {idx.feature_matrix.shape[1]}")

    cal_evidence_path = evidence_path("calibration")
    with open(cal_evidence_path, "r") as f:
        test_packets = [json.loads(next(f)) for _ in range(3)]

    for packet in test_packets:
        neighbors = idx.query(packet)
        assert all(n["transaction_id"] != packet["transaction_id"] for n in neighbors)
        assert all(n["transaction_id"] in idx.pool_ids for n in neighbors)
        print(f"\n--- {packet['case_id']} (ground_truth_is_fraud={packet['ground_truth_is_fraud']}) ---")
        print(format_exemplars_for_prompt(neighbors))
