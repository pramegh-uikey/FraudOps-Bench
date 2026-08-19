import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

DEFAULT_BAND = (0.4, 0.6)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATED_BANDS_PATH = PROJECT_ROOT / "configs" / "calibrated_bands.json"


def load_frozen_band(arm: str, path: Path = CALIBRATED_BANDS_PATH) -> tuple[float, float] | None:
    """Returns the band calibrated on the calibration split for this arm, if
    one has been frozen -- i.e. a real, once-computed calibration, not a
    live re-tune against whatever data happens to be on hand. Returns None
    if no calibration has been frozen yet for this arm (caller should fall
    back to DEFAULT_BAND and say so)."""
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    entry = data.get(arm)
    if entry is None:
        return None
    return tuple(entry["band"])


def save_frozen_band(arm: str, band: tuple[float, float], metadata: dict | None = None,
                      path: Path = CALIBRATED_BANDS_PATH) -> None:
    """Persists a calibrated band for an arm. Intended to be called exactly
    once per arm, from a calibration-split run -- never from a holdout run,
    which must consume an already-frozen band rather than producing one."""
    data = {}
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)

    entry = {
        "band": list(band),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        entry["metadata"] = metadata
    data[arm] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def probability_to_disposition(p: float, band: tuple[float, float] = DEFAULT_BAND) -> str:
    low, high = band
    if p >= high:
        return "REJECT"
    if p <= low:
        return "APPROVE"
    return "ESCALATE"


def resolve_with_second_opinion(
    p1: float,
    get_second_opinion,
    band: tuple[float, float] = DEFAULT_BAND,
    max_disagreement: float = 0.2,
) -> tuple[float, bool, float | None]:
    """Self-consistency check (Wang et al.), gated to only fire on cases the
    band would otherwise escalate -- keeps the extra call cost-bounded to
    the ambiguous subset instead of doubling every call.

    If p1 falls inside the escalate band, fetches one independent second
    estimate via get_second_opinion() (a zero-arg callable). The two
    estimates only count as agreement if they're on the same side of 0.5
    *and* within max_disagreement of each other -- same-side alone isn't
    enough (0.41 and 0.59 are technically "both sides", but 0.42 and 0.44
    should also not be waved through as confident agreement just because
    they're both < 0.5; closeness is what makes it real corroboration).
    On agreement, the returned probability is nudged just past the band
    edge on the agreed side, so a downstream probability_to_disposition()
    call commits to APPROVE/REJECT even if the raw average would still
    technically fall inside the band. On disagreement (different sides, or
    same side but too far apart), keep the original ambiguous call --
    self-consistency should confirm genuine uncertainty, not paper over it.

    Returns (final_probability, used_second_opinion, p2_or_None).
    """
    low, high = band
    if not (low < p1 < high):
        return p1, False, None

    p2 = get_second_opinion()
    same_side = (p1 < 0.5) == (p2 < 0.5)
    close_enough = abs(p1 - p2) <= max_disagreement

    if same_side and close_enough:
        avg = (p1 + p2) / 2
        nudge = 1e-6
        if avg < 0.5:
            final_p = min(avg, low - nudge)
        else:
            final_p = max(avg, high + nudge)
        return max(0.0, min(1.0, final_p)), True, p2

    return p1, True, p2


def _predicted_label(p: float, band: tuple[float, float]) -> int | None:
    disposition = probability_to_disposition(p, band)
    if disposition == "REJECT":
        return 1
    if disposition == "APPROVE":
        return 0
    return None


def risk_coverage(probabilities, ground_truth, band: tuple[float, float]) -> dict:
    """Accuracy/coverage for one band on one (probabilities, ground_truth) set --
    the standard selective-prediction evaluation pair (Chow's rule /
    Geifman & El-Yaniv): coverage = fraction of cases decided, accuracy =
    accuracy restricted to the decided subset."""
    preds = [_predicted_label(p, band) for p in probabilities]
    decided = [(p, g) for p, g in zip(preds, ground_truth) if p is not None]
    coverage = len(decided) / len(probabilities) if len(probabilities) else 0.0

    if not decided:
        return {"coverage": coverage, "accuracy": None, "n_decided": 0}

    y_pred = [p for p, _ in decided]
    y_true = [g for _, g in decided]
    return {
        "coverage": coverage,
        "accuracy": accuracy_score(y_true, y_pred),
        "n_decided": len(decided),
    }


def calibrate_band(
    probabilities,
    ground_truth,
    target_risk: float = 0.15,
    k_folds: int = 5,
    candidate_half_widths: list[float] | None = None,
    min_decided_per_band: int = 30,
    confidence_z: float = 1.645,
) -> dict:
    """Picks the narrowest symmetric ESCALATE band [0.5-h, 0.5+h] (smallest
    h => highest coverage => lowest escalate rate) whose k-fold
    cross-validated accuracy on decided cases meets (1 - target_risk).
    Chow's-rule-style risk-coverage tradeoff: fix an acceptable error rate
    on decided cases, maximize coverage subject to it.

    Selection uses a one-sided lower-confidence-bound on the CV accuracy
    (mean - confidence_z * standard_error across folds), not the raw mean.
    This was added after a real failure: on a 120-case calibration split,
    the narrowest band (h=0.05) looked like it hit 85.7% mean CV accuracy,
    but that estimate came from only ~12-13 decided cases per fold-edge --
    on a later 300-case holdout run, the same band's true accuracy was
    75.9%. The point estimate was noise dressed up as signal; the LCB
    approach correctly rejects a band whose CV estimate isn't backed by
    enough decided cases to trust, and would have selected h=0.25 instead
    of h=0.05 on that same data (LCB 0.895 vs 0.794). Also skips any
    candidate band with fewer than min_decided_per_band total decided
    cases across folds outright, regardless of how good its accuracy looks
    -- a suspiciously perfect score from 5 decided cases is not evidence.

    Falls back to the widest candidate band (most conservative, most
    escalation) if nothing meets the target risk with adequate support on
    this data.
    """
    if candidate_half_widths is None:
        candidate_half_widths = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    probabilities = np.asarray(probabilities, dtype=float)
    ground_truth = np.asarray(ground_truth, dtype=int)

    class_counts = np.bincount(ground_truth)
    min_class_count = int(class_counts[class_counts > 0].min()) if class_counts.size else 1
    n_splits = max(2, min(k_folds, min_class_count, len(probabilities)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    results = []
    for h in sorted(candidate_half_widths):
        band = (0.5 - h, 0.5 + h)
        fold_accuracies, fold_coverages, fold_n_decided = [], [], []

        for _, test_idx in skf.split(probabilities, ground_truth):
            rc = risk_coverage(probabilities[test_idx], ground_truth[test_idx], band)
            fold_coverages.append(rc["coverage"])
            fold_n_decided.append(rc["n_decided"])
            if rc["accuracy"] is not None:
                fold_accuracies.append(rc["accuracy"])

        mean_acc = float(np.mean(fold_accuracies)) if fold_accuracies else None
        if fold_accuracies and len(fold_accuracies) > 1:
            se = float(np.std(fold_accuracies, ddof=1)) / np.sqrt(len(fold_accuracies))
            lcb_acc = mean_acc - confidence_z * se
        else:
            lcb_acc = mean_acc

        results.append({
            "half_width": h,
            "band": band,
            "mean_cv_accuracy": mean_acc,
            "lcb_cv_accuracy": lcb_acc,
            "mean_cv_coverage": float(np.mean(fold_coverages)),
            "total_decided": int(sum(fold_n_decided)),
        })

    for r in sorted(results, key=lambda r: r["half_width"]):
        if (r["lcb_cv_accuracy"] is not None
                and r["lcb_cv_accuracy"] >= (1 - target_risk)
                and r["total_decided"] >= min_decided_per_band):
            return {"selected": r, "sweep": results}

    fallback = max(results, key=lambda r: r["half_width"])
    return {
        "selected": fallback,
        "sweep": results,
        "warning": (
            f"no band met target_risk={target_risk} with a {confidence_z}-sigma lower-confidence-bound "
            f"accuracy and >= {min_decided_per_band} decided cases; using widest fallback"
        ),
    }
