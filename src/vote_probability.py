"""Mechanism check: is fraud_probability discretization (Section 5.3) a
verbalization artifact, or something deeper?

Section 5.8's retrieval-augmented exemplars found real practical gains
without reducing raw probability discretization -- the motivating
hypothesis there didn't hold. This tests a more direct hypothesis: stop
asking the model to verbalize a single probability number at all (the
step that anchors it to round, prompt-shaped values), and instead sample
it N times on a simple binary question, using the vote fraction as the
probability. Naturally continuous at 1/N granularity, no verbalization
step to anchor on.

Deliberately NOT wired into run_baseline.py's arm system: no new arm in
configs/models.yaml, no calibration run, no holdout claim. This is a
standalone diagnostic on the dev split (n=50) -- explicitly the split
documented as free to use for exploratory work, never reported as a
final number (src/splits.py, paper Section 3.4) -- comparing a paired
verbalized-vs-voted probability distribution on the same cases.

See docs/future_work.md item 2 for the full rationale and sequencing
(this precedes any RL work, since RL's Brier-score reward design assumes
discretization is fixable in the first place).
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_backends import call_llm
from parsing import extract_json, strip_hidden_field
from splits import evidence_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOP_PATH = PROJECT_ROOT / "prompts" / "sop_v0.md"
N_VOTES = 10
VOTE_MAX_TOKENS = 10
VERBALIZED_MAX_TOKENS = 8192  # matches configs/models.yaml's anthropic default

RESULT_JSON_STRUCTURE = """{
  "fraud_probability": 0.0,
  "disposition": "APPROVE | ESCALATE | REJECT",
  "check_verdicts": [
    {"check": "transaction_amount_check", "verdict": "protective | risk | neutral", "detail": ""}
  ],
  "risk_indicators": [],
  "protective_indicators": [],
  "tools_used": [],
  "required_checks_completed": [],
  "missing_evidence": [],
  "evidence_used": [],
  "final_case_note": ""
}"""

FRAUD_PROBABILITY_INSTRUCTIONS = """- Fill in "check_verdicts" for all 6 required checks first, then give \
"fraud_probability" following the SOP's calibration scale -- a value near \
0.5 is correct when evidence is genuinely mixed or thin, not a failure to \
decide. Weigh convergence of multiple independent signals, not just \
whether any single risk indicator is present."""


def _make_verbalized_prompt(packet: dict, sop_text: str) -> str:
    """Identical to flows.py's _make_linear_prompt (paired baseline --
    kept as a local copy so this script has no dependency on flows.py's
    self-consistency machinery, which isn't relevant here)."""
    packet_for_agent = strip_hidden_field(packet)
    return f"""
You are a fraud analyst reviewing a card-not-present transaction alert.

Follow the SOP strictly.

SOP:
{sop_text}

CASE WITH TOOL EVIDENCE:
{json.dumps(packet_for_agent, indent=2)}

Important rules:
- You have been provided outputs from all available investigation tools.
- Use the tool evidence to complete the required checks.
- Do not mention or infer access to the hidden fraud label.
{FRAUD_PROBABILITY_INSTRUCTIONS}
- Return valid JSON only.
- Do not wrap JSON in markdown.

Return this exact JSON structure:

{RESULT_JSON_STRUCTURE}
"""


def _make_vote_prompt(packet: dict, sop_text: str) -> str:
    """Same SOP + tool-evidence sections as the verbalized prompt --
    only the final instruction changes, from 'produce structured JSON
    with a fraud_probability field' to 'answer one word.' Isolates the
    elicitation-mechanism variable."""
    packet_for_agent = strip_hidden_field(packet)
    return f"""
You are a fraud analyst reviewing a card-not-present transaction alert.

Follow the SOP strictly.

SOP:
{sop_text}

CASE WITH TOOL EVIDENCE:
{json.dumps(packet_for_agent, indent=2)}

Important rules:
- You have been provided outputs from all available investigation tools.
- Use the tool evidence to reach a judgment.
- Do not mention or infer access to the hidden fraud label.
- Weigh convergence of multiple independent signals, not just whether
  any single risk indicator is present.

Based on this evidence, is this transaction fraud or not fraud?
Respond with exactly one word, nothing else: FRAUD or NOT_FRAUD.
"""


def _parse_verbalized_probability(raw_text: str | None) -> float | None:
    if raw_text is None:
        return None
    try:
        parsed = extract_json(raw_text)
        p = parsed.get("fraud_probability")
        return float(p) if p is not None else None
    except Exception:
        return None


def _parse_vote(raw_text: str | None) -> str | None:
    """NOT_FRAUD contains FRAUD as a substring -- must check it first."""
    if raw_text is None:
        return None
    text = raw_text.strip().upper()
    if "NOT_FRAUD" in text or "NOT FRAUD" in text:
        return "NOT_FRAUD"
    if "FRAUD" in text:
        return "FRAUD"
    return None


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def run_one_case(packet: dict, sop_text: str, n_votes: int) -> dict:
    case_id = packet["case_id"]
    ground_truth = packet["ground_truth_is_fraud"]

    total_cost = 0.0
    total_latency_ms = 0.0

    verbalized_prompt = _make_verbalized_prompt(packet, sop_text)
    r = call_llm("anthropic", "claude-sonnet-5", verbalized_prompt, max_tokens=VERBALIZED_MAX_TOKENS)
    total_cost += r.cost_usd or 0.0
    total_latency_ms += r.latency_ms
    verbalized_probability = _parse_verbalized_probability(r.raw_text)
    verbalized_error = r.error

    vote_prompt = _make_vote_prompt(packet, sop_text)
    votes = []
    vote_errors = 0
    for _ in range(n_votes):
        r = call_llm("anthropic", "claude-sonnet-5", vote_prompt, max_tokens=VOTE_MAX_TOKENS)
        total_cost += r.cost_usd or 0.0
        total_latency_ms += r.latency_ms
        parsed_vote = _parse_vote(r.raw_text)
        if parsed_vote is None:
            vote_errors += 1
        votes.append(parsed_vote)

    valid_votes = [v for v in votes if v is not None]
    vote_fraction = (
        sum(1 for v in valid_votes if v == "FRAUD") / len(valid_votes)
        if valid_votes else None
    )

    return {
        "case_id": case_id,
        "ground_truth_is_fraud": ground_truth,
        "verbalized_probability": verbalized_probability,
        "verbalized_error": verbalized_error,
        "votes": votes,
        "vote_errors": vote_errors,
        "vote_fraction": vote_fraction,
        "cost_usd": total_cost,
        "latency_ms": total_latency_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="number of dev cases to run (pilot: small; full: 50)")
    parser.add_argument("--n-votes", type=int, default=N_VOTES)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    packets = load_jsonl(evidence_path("dev"))
    if args.limit is not None:
        packets = packets[: args.limit]

    sop_text = SOP_PATH.read_text()
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "vote_probability_dev.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(packets)} dev cases, {args.n_votes} votes each "
          f"({len(packets) * (1 + args.n_votes)} total API calls), concurrency={args.concurrency}")

    write_lock = threading.Lock()
    total_cost = 0.0
    start = time.monotonic()

    with open(output_path, "w") as out:
        def _write(row):
            nonlocal total_cost
            with write_lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
                total_cost += row["cost_usd"]

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(run_one_case, packet, sop_text, args.n_votes): packet
                for packet in packets
            }
            done = 0
            for future in as_completed(futures):
                row = future.result()
                _write(row)
                done += 1
                print(f"[{done}/{len(packets)}] {row['case_id']}: "
                      f"verbalized={row['verbalized_probability']}, "
                      f"vote_fraction={row['vote_fraction']}, "
                      f"vote_errors={row['vote_errors']}, "
                      f"cost=${row['cost_usd']:.4f}")

    elapsed = time.monotonic() - start
    print(f"\nDone. {len(packets)} cases in {elapsed:.0f}s. Total cost: ${total_cost:.2f}")
    print(f"Output at {output_path}")


if __name__ == "__main__":
    main()
