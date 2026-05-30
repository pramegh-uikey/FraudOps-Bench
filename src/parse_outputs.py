import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "outputs" / "evidence_baseline_gemini.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evidence_baseline_parsed.jsonl"


def extract_json(text):
    text = text.strip()

    # remove markdown fences if present
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # extract first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found")

    return json.loads(text[start:end + 1])


def main():
    parsed_rows = []

    with open(INPUT_PATH, "r") as f:
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

    with open(OUTPUT_PATH, "w") as out:
        for row in parsed_rows:
            out.write(json.dumps(row) + "\n")

    print(f"Saved parsed output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()