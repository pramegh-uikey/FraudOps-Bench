import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_PATH = PROJECT_ROOT / "outputs" / "evidence_packets_50.jsonl"
SOP_PATH = PROJECT_ROOT / "prompts" / "sop_v0.md"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evidence_baseline_qwen_local.jsonl"

MODEL_NAME = "qwen2.5:14b-instruct"

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_text(path):
    with open(path, "r") as f:
        return f.read()


def strip_hidden_fields(packet):
    packet_for_agent = dict(packet)
    packet_for_agent.pop("ground_truth_is_fraud", None)
    return packet_for_agent


def make_prompt(packet, sop_text):
    packet_for_agent = strip_hidden_fields(packet)

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


def call_local_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192
            }
        },
        timeout=900
    )

    response.raise_for_status()
    return response.json()["response"]


def main():
    packets = load_jsonl(EVIDENCE_PATH)
    sop_text = load_text(SOP_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as out:
        for idx, packet in enumerate(packets, start=1):
            print(f"Running evidence baseline {idx}/{len(packets)}: {packet['case_id']}")

            prompt = make_prompt(packet, sop_text)

            try:
                raw_response = call_local_llm(prompt)
                result = {
                    "case_id": packet["case_id"],
                    "transaction_id": packet["transaction_id"],
                    "ground_truth_is_fraud": packet["ground_truth_is_fraud"],
                    "baseline": "evidence_all_tools",
                    "model": MODEL_NAME,
                    "raw_response": raw_response
                }
            except Exception as e:
                result = {
                    "case_id": packet["case_id"],
                    "transaction_id": packet["transaction_id"],
                    "ground_truth_is_fraud": packet["ground_truth_is_fraud"],
                    "baseline": "evidence_all_tools",
                    "model": MODEL_NAME,
                    "error": str(e)
                }

            out.write(json.dumps(result) + "\n")
            out.flush()

            # free tier safe sleep; increase if still quota errors
            time.sleep(15)

    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()