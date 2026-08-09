import argparse
import json
from pathlib import Path

from parsing import extract_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PASSTHROUGH_FIELDS = [
    "arm", "flow", "backend", "model",
    "latency_ms", "input_tokens", "output_tokens", "cost_usd", "tool_call_count",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / "outputs" / f"{args.arm}.jsonl"
    output_path = PROJECT_ROOT / "outputs" / f"{args.arm}_parsed.jsonl"

    parsed_rows = []

    with open(input_path, "r") as f:
        for line in f:
            row = json.loads(line)

            parsed = {
                "case_id": row["case_id"],
                "transaction_id": row["transaction_id"],
                "ground_truth_is_fraud": row["ground_truth_is_fraud"],
                "parse_status": "not_attempted",
            }
            for field in PASSTHROUGH_FIELDS:
                if field in row:
                    parsed[field] = row[field]

            if row.get("raw_response") is None:
                parsed["parse_status"] = "no_raw_response"
                parsed["error"] = row.get("error")
                parsed_rows.append(parsed)
                continue

            try:
                llm_json = extract_json(row["raw_response"])
                parsed.update(llm_json)
                parsed["parse_status"] = "success"
            except Exception as e:
                parsed["parse_status"] = "parse_error"
                parsed["error"] = str(e)
                parsed["raw_response"] = row["raw_response"]

            parsed_rows.append(parsed)

    with open(output_path, "w") as out:
        for row in parsed_rows:
            out.write(json.dumps(row) + "\n")

    print(f"Saved parsed output to {output_path}")


if __name__ == "__main__":
    main()
