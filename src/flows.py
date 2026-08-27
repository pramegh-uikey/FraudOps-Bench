import json

from llm_backends import call_llm
from parsing import extract_json, strip_hidden_field
from flow_types import FlowResult
from selective_prediction import DEFAULT_BAND, probability_to_disposition, resolve_with_second_opinion

# run_agentic is a real LangGraph agent (brain/orchestrator/tools/memory/
# supervisor) -- see agentic_graph.py. Re-exported here so run_baseline.py's
# `from flows import run_agentic, run_direct, run_linear` keeps working
# unchanged.
from agentic_graph import run_case as run_agentic  # noqa: F401

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


def _extract_probability(raw_text: str | None) -> float | None:
    if raw_text is None:
        return None
    try:
        parsed = extract_json(raw_text)
        p = parsed.get("fraud_probability")
        return float(p) if p is not None else None
    except Exception:
        return None


def _run_completion_flow(prompt: str, backend: str, model: str, **call_kwargs) -> FlowResult:
    # Self-consistency's premise is "two independent estimates agreeing is
    # corroborating evidence" -- that only holds when there's real evidence
    # to agree about. An arm with no case-specific evidence (direct_control)
    # has nothing to differentiate cases with, so it correctly converges on
    # the SOP's neutral anchor (0.5) almost every time; two draws "agreeing"
    # on 0.5 is the model honestly reporting "no signal" twice, not a
    # confident corroborated verdict -- forcing a commit off that is
    # actively worse than leaving it ESCALATE. Default True, overridable
    # per-arm via configs/models.yaml.
    self_consistency = call_kwargs.pop("self_consistency", True)

    r = call_llm(backend, model, prompt, **call_kwargs)

    if r.error is not None or r.raw_text is None:
        return FlowResult(
            raw_response=r.raw_text,
            error=r.error,
            latency_ms=r.latency_ms,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            cost_usd=r.cost_usd,
            reasoning_tokens=r.reasoning_tokens,
            cached_input_tokens=r.cached_input_tokens,
        )

    result = FlowResult(
        raw_response=r.raw_text,
        error=None,
        latency_ms=r.latency_ms,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        cost_usd=r.cost_usd,
        reasoning_tokens=r.reasoning_tokens,
        cached_input_tokens=r.cached_input_tokens,
    )

    p1 = _extract_probability(r.raw_text)
    if p1 is None:
        # No fraud_probability field parsed (e.g. malformed JSON) -- leave
        # raw_response as-is; downstream eval falls back to the legacy
        # disposition-based scoring for this row.
        return result

    # Self-consistency second opinion, gated on the escalate band: only
    # spends an extra call when the first estimate is genuinely ambiguous.
    def _second_opinion() -> float:
        r2 = call_llm(backend, model, prompt, **call_kwargs)
        result.latency_ms += r2.latency_ms
        if r2.input_tokens is not None:
            result.input_tokens = (result.input_tokens or 0) + r2.input_tokens
        if r2.output_tokens is not None:
            result.output_tokens = (result.output_tokens or 0) + r2.output_tokens
        if r2.cost_usd is not None:
            result.cost_usd = (result.cost_usd or 0.0) + r2.cost_usd
        if r2.reasoning_tokens is not None:
            result.reasoning_tokens = (result.reasoning_tokens or 0) + r2.reasoning_tokens
        if r2.cached_input_tokens is not None:
            result.cached_input_tokens = (result.cached_input_tokens or 0) + r2.cached_input_tokens
        p2 = _extract_probability(r2.raw_text)
        return p2 if p2 is not None else p1

    if self_consistency:
        final_p, used_second_opinion, p2 = resolve_with_second_opinion(p1, _second_opinion, DEFAULT_BAND)
    else:
        final_p, used_second_opinion, p2 = p1, False, None

    try:
        parsed = extract_json(r.raw_text)
        parsed["disposition_raw"] = parsed.get("disposition")
        if used_second_opinion:
            parsed["self_consistency_check"] = {
                "triggered": True,
                "first_probability": p1,
                "second_opinion_probability": p2,
            }
            parsed["fraud_probability"] = final_p
        parsed["disposition"] = probability_to_disposition(final_p, DEFAULT_BAND)
        result.raw_response = json.dumps(parsed)
    except Exception:
        pass  # keep the original raw_response if we can't safely rewrite it

    return result


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
{FRAUD_PROBABILITY_INSTRUCTIONS}
- Return valid JSON only.
- Do not wrap JSON in markdown.

Return this exact JSON structure:

{RESULT_JSON_STRUCTURE}
"""


def run_direct(case: dict, sop_text: str, backend: str, model: str, **call_kwargs) -> FlowResult:
    return _run_completion_flow(_make_direct_prompt(case, sop_text), backend, model, **call_kwargs)


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
{FRAUD_PROBABILITY_INSTRUCTIONS}
- Return valid JSON only.
- Do not wrap JSON in markdown.

Return this exact JSON structure:

{RESULT_JSON_STRUCTURE}
"""


def run_linear(packet: dict, sop_text: str, backend: str, model: str, **call_kwargs) -> FlowResult:
    return _run_completion_flow(_make_linear_prompt(packet, sop_text), backend, model, **call_kwargs)


# run_agentic is imported from agentic_graph above (see top of file).
