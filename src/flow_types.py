from dataclasses import dataclass


@dataclass
class FlowResult:
    raw_response: str | None
    error: str | None
    tool_trace: list | None = None
    tool_call_count: int | None = None
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    # OpenAI-specific (None for Anthropic/Ollama) -- see LLMCallResult in
    # llm_backends.py for what these track and why.
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None
