import os
import json
import time
from pathlib import Path
from google import genai
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES_PATH = PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v0_cases.jsonl"
SOP_PATH = PROJECT_ROOT / "prompts" / "sop_v0.md"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "direct_baseline_gemini.jsonl"

MODEL_NAME = "gemini-3.5-flash"


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_text(path):
    with open(path, "r") as f:
        return f.read()


def strip_hidden_fields(case):
    case_for_agent = dict(case)
    case_for_agent.pop("ground_truth_is_fraud", None)
    return case_for_agent


def make_prompt(case, sop_text):
    case_for_agent = strip_hidden_fields(case)

    return f"""
You are a fraud analyst reviewing a card-not-present transaction alert.

Follow the SOP strictly.

SOP:
{sop_text}

CASE:
{json.dumps(case_for_agent, indent=2)}

Important rules:
- You do not have access to investigation tools in this baseline.
- Use only the visible case summary.
- Do not mention the hidden fraud label.
- Return valid JSON only.
- Do not wrap JSON in markdown.

Return this exact JSON structure:

{{
  "disposition": "APPROVE | ESCALATE | REJECT",
  "risk_indicators": [],
  "protective_indicators": [],
  "tools_used": [],
  "required_checks_completed": [],
  "missing_evidence": [],
  "evidence_used": [],
  "final_case_note": ""
}}
"""


def call_gemini(prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text


def main():
    cases = load_jsonl(CASES_PATH)
    sop_text = load_text(SOP_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as out:
        for idx, case in enumerate(cases, start=1):
            print(f"Running case {idx}/{len(cases)}: {case['case_id']}")

            prompt = make_prompt(case, sop_text)

            try:
                raw_response = call_gemini(prompt)
                result = {
                    "case_id": case["case_id"],
                    "transaction_id": case["transaction_id"],
                    "ground_truth_is_fraud": case["ground_truth_is_fraud"],
                    "baseline": "direct_no_tools",
                    "model": MODEL_NAME,
                    "raw_response": raw_response
                }
            except Exception as e:
                result = {
                    "case_id": case["case_id"],
                    "transaction_id": case["transaction_id"],
                    "ground_truth_is_fraud": case["ground_truth_is_fraud"],
                    "baseline": "direct_no_tools",
                    "model": MODEL_NAME,
                    "error": str(e)
                }

            out.write(json.dumps(result) + "\n")
            out.flush()

            time.sleep(1)

    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()