import json
import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found")

    return json.loads(text[start:end + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["direct", "evidence"])
    args = parser.parse_args()

    if args.baseline == "direct":
        input_path = PROJECT_ROOT / "outputs" / "direct_baseline_gemini.jsonl"
        output_path = PROJECT_ROOT / "outputs" / "direct_baseline_parsed.jsonl"
    else:
        input_path = PROJECT_ROOT / "outputs" / "evidence_baseline_gemini.jsonl"
        output_path = PROJECT_ROOT / "outputs" / "evidence_baseline_parsed.jsonl"

    parsed_rows = []

    with open(input_path, "r") as f:
        for line in f:
            row = json.loads(line)

            parsed = {
                "case_id": row["case_id"],
                "transaction_id": row["transaction_id"],
                "ground_truth_is_fraud": row["ground_truth_is_fraud"],
                "baseline": row["baseline"],
                "model": row["model"],
                "parse_status": "not_attempted"
            }

            if "raw_response" not in row:
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