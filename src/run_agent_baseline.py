import os
import json
import time
import re
from pathlib import Path
from google import genai

from tools import (
    get_transaction_details,
    get_card_history,
    get_email_domain_profile,
    get_device_history,
    get_velocity_summary,
    get_identity_match_summary,
)

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES_PATH = PROJECT_ROOT / "data" / "processed" / "fraudops_bench_v0_cases.jsonl"
SOP_PATH = PROJECT_ROOT / "prompts" / "sop_v0.md"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "agent_baseline_gemini.jsonl"

MODEL_NAME = "gemini-3.5-flash"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


TOOL_REGISTRY = {
    "get_transaction_details": get_transaction_details,
    "get_card_history": get_card_history,
    "get_email_domain_profile": get_email_domain_profile,
    "get_device_history": get_device_history,
    "get_velocity_summary": get_velocity_summary,
    "get_identity_match_summary": get_identity_match_summary,
}


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_text(path):
    with open(path, "r") as f:
        return f.read()


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


def call_gemini(prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text


def strip_hidden_case_fields(case):
    case_for_agent = dict(case)
    case_for_agent.pop("ground_truth_is_fraud", None)
    return case_for_agent


def make_tool_selection_prompt(case, sop_text, tool_trace):
    case_for_agent = strip_hidden_case_fields(case)

    return f"""
You are a fraud analyst agent.

You must investigate the transaction by selecting one tool at a time.

SOP:
{sop_text}

CASE:
{json.dumps(case_for_agent, indent=2)}

Available tools:
- get_transaction_details
- get_card_history
- get_email_domain_profile
- get_device_history
- get_velocity_summary
- get_identity_match_summary

Tool trace so far:
{json.dumps(tool_trace, indent=2)}

Instructions:
- Choose the next most useful tool.
- Do not repeat a tool already used unless absolutely necessary.
- If enough evidence has been gathered, choose FINAL_DECISION.
- Return valid JSON only.
- Do not wrap in markdown.

Return exactly one of these:

For tool call:
{{
  "action": "CALL_TOOL",
  "tool_name": "get_transaction_details | get_card_history | get_email_domain_profile | get_device_history | get_velocity_summary | get_identity_match_summary",
  "reason": ""
}}

For final decision:
{{
  "action": "FINAL_DECISION",
  "reason": ""
}}
"""


def make_final_decision_prompt(case, sop_text, tool_trace):
    case_for_agent = strip_hidden_case_fields(case)

    return f"""
You are a fraud analyst.

Use the SOP, the case summary, and the gathered tool evidence to make a final decision.

SOP:
{sop_text}

CASE:
{json.dumps(case_for_agent, indent=2)}

TOOL TRACE:
{json.dumps(tool_trace, indent=2)}

Important:
- Do not mention or infer access to the hidden fraud label.
- Use only the case summary and tool evidence.
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


def run_agent_for_case(case, sop_text, max_steps=6):
    transaction_id = case["transaction_id"]
    tool_trace = []
    tools_used = set()

    for step in range(1, max_steps + 1):
        prompt = make_tool_selection_prompt(case, sop_text, tool_trace)
        raw_action_response = call_gemini(prompt)

        try:
            action_json = extract_json(raw_action_response)
        except Exception as e:
            tool_trace.append({
                "step": step,
                "type": "parse_error",
                "raw_response": raw_action_response,
                "error": str(e)
            })
            break

        action = action_json.get("action")

        if action == "FINAL_DECISION":
            tool_trace.append({
                "step": step,
                "type": "final_decision_requested",
                "reason": action_json.get("reason")
            })
            break

        if action != "CALL_TOOL":
            tool_trace.append({
                "step": step,
                "type": "invalid_action",
                "action_json": action_json
            })
            break

        tool_name = action_json.get("tool_name")

        if tool_name not in TOOL_REGISTRY:
            tool_trace.append({
                "step": step,
                "type": "invalid_tool",
                "tool_name": tool_name,
                "action_json": action_json
            })
            break

        if tool_name in tools_used:
            tool_trace.append({
                "step": step,
                "type": "repeated_tool",
                "tool_name": tool_name,
                "reason": action_json.get("reason")
            })
            continue

        tool_output = TOOL_REGISTRY[tool_name](transaction_id)
        tools_used.add(tool_name)

        tool_trace.append({
            "step": step,
            "type": "tool_call",
            "tool_name": tool_name,
            "reason": action_json.get("reason"),
            "tool_output": tool_output
        })

    final_prompt = make_final_decision_prompt(case, sop_text, tool_trace)
    raw_final_response = call_gemini(final_prompt)

    return {
        "tool_trace": tool_trace,
        "raw_response": raw_final_response
    }


def load_completed_case_ids(output_path):
    completed = set()

    if not output_path.exists():
        return completed

    with open(output_path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "raw_response" in row:
                    completed.add(row["case_id"])
            except Exception:
                continue

    return completed


def main():
    cases = load_jsonl(CASES_PATH)
    sop_text = load_text(SOP_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_case_ids(OUTPUT_PATH)

    with open(OUTPUT_PATH, "a") as out:
        for idx, case in enumerate(cases, start=1):
            case_id = case["case_id"]

            if case_id in completed:
                print(f"Skipping completed case {case_id}")
                continue

            print(f"Running agent baseline {idx}/{len(cases)}: {case_id}")

            try:
                result_data = run_agent_for_case(case, sop_text)

                result = {
                    "case_id": case["case_id"],
                    "transaction_id": case["transaction_id"],
                    "ground_truth_is_fraud": case["ground_truth_is_fraud"],
                    "baseline": "agent_tool_select",
                    "model": MODEL_NAME,
                    "tool_trace": result_data["tool_trace"],
                    "raw_response": result_data["raw_response"]
                }

            except Exception as e:
                result = {
                    "case_id": case["case_id"],
                    "transaction_id": case["transaction_id"],
                    "ground_truth_is_fraud": case["ground_truth_is_fraud"],
                    "baseline": "agent_tool_select",
                    "model": MODEL_NAME,
                    "error": str(e)
                }

            out.write(json.dumps(result) + "\n")
            out.flush()

            time.sleep(15)

    print(f"Saved agent baseline results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()