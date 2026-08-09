import json

from llm_backends import call_llm
from parsing import strip_hidden_field
from flow_types import FlowResult

# run_agentic is a real LangGraph agent (brain/orchestrator/tools/memory/
# supervisor) -- see agentic_graph.py. Re-exported here so run_baseline.py's
# `from flows import run_agentic, run_direct, run_linear` keeps working
# unchanged.
from agentic_graph import run_case as run_agentic  # noqa: F401

RESULT_JSON_STRUCTURE = """{
  "disposition": "APPROVE | ESCALATE | REJECT",
  "risk_indicators": [],
  "protective_indicators": [],
  "tools_used": [],
  "required_checks_completed": [],
  "missing_evidence": [],
  "evidence_used": [],
  "final_case_note": ""
}"""


def _make_direct_prompt(case: dict, sop_text: str) -> str:
    case_for_agent = strip_hidden_field(case)

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

{RESULT_JSON_STRUCTURE}
"""


def run_direct(case: dict, sop_text: str, backend: str, model: str, **call_kwargs) -> FlowResult:
    prompt = _make_direct_prompt(case, sop_text)
    r = call_llm(backend, model, prompt, **call_kwargs)

    return FlowResult(
        raw_response=r.raw_text,
        error=r.error,
        latency_ms=r.latency_ms,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cost_usd=r.cost_usd,
    )


def _make_linear_prompt(packet: dict, sop_text: str) -> str:
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
- Return valid JSON only.
- Do not wrap JSON in markdown.

Return this exact JSON structure:

{RESULT_JSON_STRUCTURE}
"""


def run_linear(packet: dict, sop_text: str, backend: str, model: str, **call_kwargs) -> FlowResult:
    prompt = _make_linear_prompt(packet, sop_text)
    r = call_llm(backend, model, prompt, **call_kwargs)

    return FlowResult(
        raw_response=r.raw_text,
        error=r.error,
        latency_ms=r.latency_ms,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cost_usd=r.cost_usd,
    )


# run_agentic is imported from agentic_graph above (see top of file).
