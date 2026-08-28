from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS = ("dev", "calibration", "holdout", "holdout_v2", "retrieval_pool")

_CASES_PATH = {
    "dev": PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v0_cases.jsonl",
    "calibration": PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v1_calibration_cases.jsonl",
    "holdout": PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v1_holdout_cases.jsonl",
    # holdout_v2: the original holdout run used a calibrate_band() that
    # turned out to overfit small-sample noise (see docs/methodology_log.md,
    # 2026-08-16 entries). holdout (v1) is "spent" -- its labels were used
    # to diagnose and validate the fix, so it can't be reused as a clean
    # single-use holdout anymore. holdout_v2 is a fresh, disjoint sample for
    # a genuinely unpeeked re-check under the corrected methodology.
    "holdout_v2": PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v2_holdout_cases.jsonl",
    # retrieval_pool: a dedicated, disjoint pool of labeled cases used only
    # as the k-NN exemplar source for linear_retrieval/agentic_retrieval.
    # Never scored itself, never reused as another split's cases -- kept
    # out of load_excluded_transaction_ids()'s union like every other
    # split, so it can never leak into a future split either.
    "retrieval_pool": PROJECT_ROOT / "data" / "processed" / "fraudops_bench_retrieval_pool_cases.jsonl",
}


def _check_split(split: str):
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}', expected one of {SPLITS}")


def cases_path(split: str) -> Path:
    _check_split(split)
    return _CASES_PATH[split]


def evidence_path(split: str) -> Path:
    _check_split(split)
    # dev keeps its original, pre-split-aware filename/location so existing
    # dev-set outputs and any pointers to them stay valid unchanged.
    if split == "dev":
        return PROJECT_ROOT / "outputs" / "evidence_packets_50.jsonl"
    return PROJECT_ROOT / "outputs" / split / "evidence_packets.jsonl"


def outputs_dir(split: str) -> Path:
    _check_split(split)
    if split == "dev":
        return PROJECT_ROOT / "outputs"
    return PROJECT_ROOT / "outputs" / split


def arm_output_path(split: str, arm: str, run_tag: str | None = None, suffix: str = ".jsonl") -> Path:
    """Path convention for a run's raw output file. dev + no run_tag matches
    today's exact `outputs/{arm}.jsonl` layout; anything else nests under
    outputs/{split}/ and appends _{run_tag} for repeat runs."""
    name = arm if run_tag is None else f"{arm}_{run_tag}"
    return outputs_dir(split) / f"{name}{suffix}"
