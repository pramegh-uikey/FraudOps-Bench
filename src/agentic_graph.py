import json
import os
import time
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from flow_types import FlowResult
from llm_backends import ANTHROPIC_COST_PER_MTOK
from parsing import strip_hidden_field
from tools import (
    get_transaction_details as _get_transaction_details,
    get_card_history as _get_card_history,
    get_email_domain_profile as _get_email_domain_profile,
    get_device_history as _get_device_history,
    get_velocity_summary as _get_velocity_summary,
    get_identity_match_summary as _get_identity_match_summary,
)

load_dotenv()

DEFAULT_MAX_RETRIES = 5


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    memory: list[dict]
    step_count: int
    next_action: str
    final_disposition: dict | None


class SupervisorDecision(BaseModel):
    action: Literal["continue", "finalize"] = Field(
        description="whether the investigation needs more evidence or is ready for a final disposition"
    )
    unmet_checks: list[str] = Field(
        default_factory=list, description="required checks from the SOP not yet addressed by gathered evidence"
    )
    guidance: str = Field(
        default="", description="one short instruction for what to investigate next; empty if finalizing"
    )


class Disposition(BaseModel):
    disposition: Literal["APPROVE", "ESCALATE", "REJECT"]
    risk_indicators: list[str] = Field(default_factory=list)
    protective_indicators: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    required_checks_completed: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    final_case_note: str = ""


BRAIN_SYSTEM_PROMPT = """You are a fraud analyst agent investigating a card-not-present transaction alert.

Follow the SOP strictly. Use the available tools to gather the evidence needed \
for each required check, one tool call at a time. When you have gathered \
enough evidence to complete all required checks, respond in plain text \
summarizing what you found instead of calling another tool -- do not \
fabricate a tool call just to stop.

Do not mention or infer access to any hidden fraud label.

SOP:
{sop_text}

CASE:
{case_json}
"""

SUPERVISOR_PROMPT = """You are the supervisor overseeing a fraud investigation in progress. \
You do not gather evidence yourself -- you review the analyst's progress \
against the SOP's required checks and decide whether to send them back for \
more evidence or approve moving to a final decision.

Required checks from the SOP:
{required_checks}

Evidence gathered so far (structured investigation memory):
{memory_json}

Decide: has enough evidence been gathered to address every required check? \
If not, name the unmet checks and give one short, concrete instruction for \
what to investigate next.
"""

FINALIZE_PROMPT = """You are a fraud analyst finalizing your assessment of a card-not-present \
transaction alert.

Follow the SOP strictly. Use only the case summary and the evidence gathered \
below. Do not mention or infer access to any hidden fraud label.

SOP:
{sop_text}

CASE:
{case_json}

EVIDENCE GATHERED (structured investigation memory):
{memory_json}

Produce the final disposition.
"""


def _get_chat_model(backend: str, model: str, **call_kwargs):
    if backend == "anthropic":
        max_tokens = call_kwargs.get("max_tokens", 4096)
        # NOTE: temperature/top_p/top_k intentionally never set -- claude-sonnet-5
        # and claude-opus-5 return 400 on non-default values (same constraint as
        # llm_backends.py's Anthropic path).
        return ChatAnthropic(
            model=model,
            max_tokens=max_tokens,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    if backend == "ollama":
        return ChatOllama(
            model=model,
            temperature=call_kwargs.get("temperature", 0.0),
            num_ctx=call_kwargs.get("num_ctx", 8192),
            base_url=call_kwargs.get("base_url", "http://localhost:11434"),
        )
    raise ValueError(f"Unknown backend '{backend}'")


def _make_case_tools(transaction_id):
    """Wraps the 6 tools.py functions as LangChain tools, closed over this
    case's transaction_id so the model never has to supply it. Tool names
    match the originals so tools_used stays comparable across arms."""

    @tool
    def get_transaction_details() -> dict:
        """Get the core details of the current transaction: amount, product
        code, card brand/type, purchaser/recipient email domains, device
        info, and match flags."""
        return _get_transaction_details(transaction_id)

    @tool
    def get_card_history() -> dict:
        """Get prior-transaction history for the card used in this
        transaction: prior transaction count, confirmed fraud count and
        rate, and amount pattern."""
        return _get_card_history(transaction_id)

    @tool
    def get_email_domain_profile() -> dict:
        """Get fraud-rate history for the purchaser and recipient email
        domains on this transaction, and whether the domains match."""
        return _get_email_domain_profile(transaction_id)

    @tool
    def get_device_history() -> dict:
        """Get prior-transaction history for the device used in this
        transaction: prior transaction count, confirmed fraud count and
        rate, and amount pattern."""
        return _get_device_history(transaction_id)

    @tool
    def get_velocity_summary() -> dict:
        """Get anonymized count and time-delta feature summaries used to
        detect unusually high transaction velocity."""
        return _get_velocity_summary(transaction_id)

    @tool
    def get_identity_match_summary() -> dict:
        """Get device/browser/OS/screen and identity match-flag consistency
        signals for this transaction."""
        return _get_identity_match_summary(transaction_id)

    return [
        get_transaction_details,
        get_card_history,
        get_email_domain_profile,
        get_device_history,
        get_velocity_summary,
        get_identity_match_summary,
    ]


def _usage_cost(usage_metadata: dict | None, backend: str, model: str) -> tuple[int | None, int | None, float | None]:
    if not usage_metadata:
        return None, None, None
    input_tokens = usage_metadata.get("input_tokens")
    output_tokens = usage_metadata.get("output_tokens")
    cost_usd = None
    if backend == "anthropic" and model in ANTHROPIC_COST_PER_MTOK and input_tokens is not None:
        in_rate, out_rate = ANTHROPIC_COST_PER_MTOK[model]
        cost_usd = (input_tokens * in_rate + (output_tokens or 0) * out_rate) / 1_000_000
    elif backend == "ollama":
        cost_usd = 0.0
    return input_tokens, output_tokens, cost_usd


def build_graph(
    case: dict,
    sop_text: str,
    backend: str,
    model: str,
    max_steps: int,
    usage_log: list,
    **call_kwargs,
):
    """Builds and compiles the 4-node agentic graph for one case. Rebuilt
    per case since tools are closed over this case's transaction_id and the
    SOP/case text are baked into the node prompts -- the cost of doing so is
    negligible next to the LLM round trips it wraps."""

    transaction_id = case["transaction_id"]
    case_for_agent = strip_hidden_field(case)
    case_json = json.dumps(case_for_agent, indent=2)
    required_checks = json.dumps(case.get("required_checks", []))

    max_retries = call_kwargs.get("max_retries", DEFAULT_MAX_RETRIES)

    case_tools = _make_case_tools(transaction_id)
    llm_with_tools = (
        _get_chat_model(backend, model, **call_kwargs)
        .bind_tools(case_tools)
        .with_retry(stop_after_attempt=max_retries)
    )

    supervisor_llm = (
        _get_chat_model(backend, model, **call_kwargs)
        .with_structured_output(SupervisorDecision, include_raw=True)
        .with_retry(stop_after_attempt=max_retries)
    )
    finalize_llm = (
        _get_chat_model(backend, model, **call_kwargs)
        .with_structured_output(Disposition, include_raw=True)
        .with_retry(stop_after_attempt=max_retries)
    )

    def _log_usage(raw_message) -> None:
        usage = getattr(raw_message, "usage_metadata", None)
        input_tokens, output_tokens, cost_usd = _usage_cost(usage, backend, model)
        usage_log.append({"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd})

    def brain_node(state: AgentState) -> dict:
        system = SystemMessage(content=BRAIN_SYSTEM_PROMPT.format(sop_text=sop_text, case_json=case_json))
        # Anthropic rejects a request with only a system message (needs a
        # leading user turn); on the very first call state["messages"] is
        # empty, so seed a kickoff turn. Ollama is lenient here, but this
        # keeps both backends on the same, spec-correct message shape.
        history = state["messages"] or [HumanMessage(content="Begin the investigation.")]
        response = llm_with_tools.invoke([system] + history)
        _log_usage(response)
        return {"messages": [response], "step_count": state["step_count"] + 1}

    def tools_node(state: AgentState) -> dict:
        raw_tool_node = ToolNode(case_tools)
        result = raw_tool_node.invoke(state)
        new_messages = result["messages"]
        memory_additions = [
            {"tool_name": m.name, "step": state["step_count"], "output": m.content}
            for m in new_messages
        ]
        return {"messages": new_messages, "memory": state["memory"] + memory_additions}

    def supervisor_node(state: AgentState) -> dict:
        if state["step_count"] >= max_steps:
            return {"next_action": "finalize"}

        prompt = SUPERVISOR_PROMPT.format(
            required_checks=required_checks,
            memory_json=json.dumps(state["memory"], indent=2),
        )
        result = supervisor_llm.invoke(prompt)
        _log_usage(result["raw"])
        decision = result["parsed"]

        if decision is None or decision.action == "finalize":
            return {"next_action": "finalize"}

        guidance_msg = HumanMessage(
            content=f"[Supervisor] Unmet checks: {decision.unmet_checks}. {decision.guidance}"
        )
        return {"messages": [guidance_msg], "next_action": "continue"}

    def finalize_node(state: AgentState) -> dict:
        prompt = FINALIZE_PROMPT.format(
            sop_text=sop_text,
            case_json=case_json,
            memory_json=json.dumps(state["memory"], indent=2),
        )
        result = finalize_llm.invoke(prompt)
        _log_usage(result["raw"])
        disposition = result["parsed"]

        if disposition is None:
            return {
                "final_disposition": None,
                "messages": [AIMessage(content="")],
            }

        return {
            "final_disposition": disposition.model_dump(),
            "messages": [AIMessage(content=json.dumps(disposition.model_dump()))],
        }

    def route_after_brain(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "supervisor"

    def route_after_supervisor(state: AgentState) -> str:
        return "brain" if state["next_action"] == "continue" else "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("brain", brain_node)
    graph.add_node("tools", tools_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "brain")
    graph.add_conditional_edges("brain", route_after_brain, {"tools": "tools", "supervisor": "supervisor"})
    graph.add_edge("tools", "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, {"brain": "brain", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()


def run_case(case: dict, sop_text: str, backend: str, model: str, max_steps: int = 6, **call_kwargs) -> FlowResult:
    usage_log: list = []
    start = time.monotonic()

    compiled = build_graph(case, sop_text, backend, model, max_steps, usage_log, **call_kwargs)

    try:
        final_state = compiled.invoke(
            {"messages": [], "memory": [], "step_count": 0, "next_action": "", "final_disposition": None},
            config={"recursion_limit": max_steps * 4 + 10},
        )
    except Exception as e:
        return FlowResult(raw_response=None, error=str(e), latency_ms=(time.monotonic() - start) * 1000)

    latency_ms = (time.monotonic() - start) * 1000

    total_input_tokens = sum(u["input_tokens"] for u in usage_log if u["input_tokens"] is not None) or None
    total_output_tokens = sum(u["output_tokens"] for u in usage_log if u["output_tokens"] is not None) or None
    total_cost = sum(u["cost_usd"] for u in usage_log if u["cost_usd"] is not None) if usage_log else None

    disposition = final_state.get("final_disposition")
    tool_call_count = sum(1 for m in final_state["memory"])

    if disposition is None:
        return FlowResult(
            raw_response=None,
            error="agentic_graph: finalize node failed to produce a valid structured disposition",
            tool_trace=final_state["memory"],
            tool_call_count=tool_call_count,
            latency_ms=latency_ms,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=total_cost,
        )

    return FlowResult(
        raw_response=json.dumps(disposition),
        error=None,
        tool_trace=final_state["memory"],
        tool_call_count=tool_call_count,
        latency_ms=latency_ms,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=total_cost,
    )
