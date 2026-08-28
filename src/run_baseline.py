import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from config import get_arm_config, load_models_config
from flows import run_agentic, run_direct, run_linear
from freeze_methodology import verify_manifest
from splits import arm_output_path, cases_path, evidence_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOP_PATH = PROJECT_ROOT / "prompts" / "sop_v0.md"

FLOW_RUNNERS = {
    "direct": run_direct,
    "linear": run_linear,
    "agentic": run_agentic,
}

CALL_KWARG_KEYS = {
    "max_tokens", "temperature", "num_ctx", "base_url",
    "max_retries", "base_delay_s", "max_delay_s",
    "self_consistency", "use_retrieval",
}


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_text(path):
    with open(path, "r") as f:
        return f.read()


def load_existing_rows(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    rows = []
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--split", choices=["dev", "calibration", "holdout", "holdout_v2"], default="dev")
    parser.add_argument("--run-tag", default=None,
                         help="tag appended to the output filename, for repeat runs on the same split "
                              "(variance estimation) without overwriting a previous run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-ids", default=None,
                         help="comma-separated case_id list to run instead of --limit")
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true",
                         help="bypass the holdout frozen-methodology guard (loud, logged override)")
    parser.add_argument("--override-self-consistency", choices=["true", "false"], default=None,
                         help="force self_consistency on/off regardless of configs/models.yaml, "
                              "for ablation runs (e.g. --run-tag no_sc)")
    args = parser.parse_args()

    is_holdout_split = args.split in ("holdout", "holdout_v2")
    if is_holdout_split and not args.force:
        ok, message = verify_manifest()
        if not ok:
            raise SystemExit(
                f"Refusing to run --split {args.split}: {message}\n"
                "Holdout sets are single-use for final reporting -- they must only be run "
                "against a methodology that's been frozen with freeze_methodology.py, so the "
                "results can't be quietly re-tuned after a peek. Pass --force to override "
                "(this will be obvious in the run, not silent)."
            )
        print(f"Holdout methodology check: {message}")
    elif is_holdout_split and args.force:
        print(f"!!! --force: bypassing the frozen-methodology guard for a {args.split} run. "
              "This run's results are NOT protected against post-hoc tuning leakage. !!!")

    config = load_models_config(args.config) if args.config else load_models_config()
    arm_config = get_arm_config(args.arm, config)

    flow = arm_config["flow"]
    if flow == "classical_ml":
        raise SystemExit(
            "classical_ml is not run through run_baseline.py — use "
            "train_classical_baseline.py and run_classical_baseline.py instead."
        )
    if flow not in FLOW_RUNNERS:
        raise SystemExit(f"Unknown flow '{flow}' for arm '{args.arm}'")

    backend = arm_config["backend"]
    model = arm_config["model"]
    call_kwargs = {k: v for k, v in arm_config.items() if k in CALL_KWARG_KEYS}
    if args.override_self_consistency is not None:
        call_kwargs["self_consistency"] = args.override_self_consistency == "true"
        print(f"Overriding self_consistency={call_kwargs['self_consistency']} for this run (ablation).")

    output_path = Path(args.output) if args.output else arm_output_path(args.split, args.arm, args.run_tag)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sop_text = load_text(SOP_PATH)

    if flow == "linear":
        items = load_jsonl(evidence_path(args.split))
    else:
        items = load_jsonl(cases_path(args.split))

    if args.case_ids:
        wanted = set(args.case_ids.split(","))
        items = [item for item in items if item["case_id"] in wanted]
    elif args.limit is not None:
        items = items[: args.limit]

    target_case_ids = {item["case_id"] for item in items}

    existing_rows = load_existing_rows(output_path)
    completed_ids = {
        row["case_id"] for row in existing_rows
        if row.get("raw_response") is not None
    }
    # Keep successful rows, and any row for a case outside this run's target
    # set untouched. Stale error rows for cases we're about to retry are
    # dropped here so a retry doesn't pile up duplicate error rows.
    kept_rows = [
        row for row in existing_rows
        if row["case_id"] in completed_ids or row["case_id"] not in target_case_ids
    ]
    with open(output_path, "w") as out:
        for row in kept_rows:
            out.write(json.dumps(row) + "\n")

    to_run = [item for item in items if item["case_id"] not in completed_ids]

    print(f"Arm '{args.arm}': {len(target_case_ids)} targeted, "
          f"{len(completed_ids & target_case_ids)} already complete, {len(to_run)} to run.")

    with open(output_path, "a") as out:
        for item in tqdm(to_run, desc=args.arm):
            case_id = item["case_id"]
            transaction_id = item["transaction_id"]
            ground_truth = item["ground_truth_is_fraud"]

            runner = FLOW_RUNNERS[flow]
            if flow == "linear":
                result = runner(item, sop_text, backend, model, **call_kwargs)
            else:
                result = runner(item, sop_text, backend, model, **call_kwargs)

            row = {
                "case_id": case_id,
                "transaction_id": transaction_id,
                "ground_truth_is_fraud": ground_truth,
                "arm": args.arm,
                "flow": flow,
                "backend": backend,
                "model": model,
                "raw_response": result.raw_response,
                "error": result.error,
                "tool_trace": result.tool_trace,
                "tool_call_count": result.tool_call_count,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "reasoning_tokens": result.reasoning_tokens,
                "cached_input_tokens": result.cached_input_tokens,
            }

            out.write(json.dumps(row) + "\n")
            out.flush()

    print(f"Done. Output at {output_path}")


if __name__ == "__main__":
    main()
