import os
import random
import time
from dataclasses import dataclass, field

import anthropic
import openai
import requests
from dotenv import load_dotenv

load_dotenv()

# $ per million tokens: (input, output)
ANTHROPIC_COST_PER_MTOK = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# $ per million tokens: (input, cached_input, cache_write, output).
# Verified 2026-08-28 via developers.openai.com/api/docs/models/gpt-5.6-terra
# and cross-checked against a live chat.completions.create call's usage
# object (prompt_tokens_details.cached_tokens / cache_write_tokens,
# completion_tokens_details.reasoning_tokens all confirmed present).
# Note: OpenAI doubles input / 1.5x's output pricing for prompts over 272K
# input tokens -- not relevant here (our prompts run well under that).
OPENAI_COST_PER_MTOK = {
    "gpt-5.6-terra": (2.00, 0.20, 2.50, 12.00),
}

_anthropic_client = None
_openai_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


@dataclass
class LLMCallResult:
    raw_text: str | None
    error: str | None
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    stop_reason: str | None = None
    attempts: int = 1
    # OpenAI-specific (None for Anthropic/Ollama): reasoning-tier models bill
    # invisible reasoning tokens as part of output_tokens, and can spend an
    # entire max_completion_tokens budget on reasoning with zero visible text
    # (confirmed live, 2026-08-28: a 20-token cap produced finish_reason=
    # "length", content="", reasoning_tokens=20/20). Tracked separately here
    # so that failure mode is diagnosable rather than looking like a generic
    # empty response. cached_input_tokens is the cache-read count (billed at
    # a 90% discount vs. the base input rate) -- kept separate from
    # input_tokens (which OpenAI reports inclusive of cached tokens) so cost
    # math can apply the discounted rate correctly.
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None


_ANTHROPIC_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)
_ANTHROPIC_RETRYABLE_STATUS = {429, 500, 502, 503, 529}


def _with_retry(call_fn, max_retries: int, base_delay_s: float, max_delay_s: float,
                 is_retryable_error) -> LLMCallResult:
    attempt = 0
    last_error = None

    while attempt < max_retries:
        attempt += 1
        try:
            result = call_fn()
            result.attempts = attempt
            return result
        except Exception as e:
            if not is_retryable_error(e) or attempt >= max_retries:
                return LLMCallResult(
                    raw_text=None,
                    error=str(e),
                    latency_ms=0.0,
                    attempts=attempt,
                )
            last_error = e
            delay = min(base_delay_s * (2 ** (attempt - 1)) + random.uniform(0, 1), max_delay_s)
            time.sleep(delay)

    return LLMCallResult(raw_text=None, error=str(last_error), latency_ms=0.0, attempts=attempt)


def _is_anthropic_retryable(e: Exception) -> bool:
    if isinstance(e, _ANTHROPIC_RETRYABLE):
        return True
    if isinstance(e, anthropic.APIStatusError):
        return e.status_code in _ANTHROPIC_RETRYABLE_STATUS
    return False


def call_anthropic(
    prompt: str,
    model: str,
    max_tokens: int = 2048,
    system: str | None = None,
    max_retries: int = 5,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    **_ignored,
) -> LLMCallResult:
    client = _get_anthropic_client()

    def _call() -> LLMCallResult:
        start = time.monotonic()
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        # NOTE: temperature/top_p/top_k are intentionally never set here.
        # claude-sonnet-5 / claude-opus-5 return a 400 on non-default values.
        response = client.messages.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        text = next((b.text for b in response.content if b.type == "text"), None)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = None
        if model in ANTHROPIC_COST_PER_MTOK:
            in_rate, out_rate = ANTHROPIC_COST_PER_MTOK[model]
            cost_usd = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000

        # A response with no text block (e.g. hit max_tokens before any text
        # was emitted) must surface as an error, not a silent raw_text=None
        # success -- otherwise it's undiagnosable and looks identical to a
        # genuine empty-response bug.
        error = None if text is not None else (
            f"Anthropic response contained no text block (stop_reason={response.stop_reason})"
        )

        return LLMCallResult(
            raw_text=text,
            error=error,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            stop_reason=response.stop_reason,
        )

    return _with_retry(_call, max_retries, base_delay_s, max_delay_s, _is_anthropic_retryable)


_OPENAI_RETRYABLE = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
)
_OPENAI_RETRYABLE_STATUS = {429, 500, 502, 503, 529}


def _is_openai_retryable(e: Exception) -> bool:
    if isinstance(e, _OPENAI_RETRYABLE):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code in _OPENAI_RETRYABLE_STATUS
    return False


def call_openai(
    prompt: str,
    model: str,
    max_tokens: int = 2048,
    system: str | None = None,
    max_retries: int = 5,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    **_ignored,
) -> LLMCallResult:
    # Verified live against gpt-5.6-terra, 2026-08-28 (see phase_a3_shapecheck.py
    # in that day's scratchpad for the raw transcripts):
    #   - Chat Completions API (client.chat.completions.create) -- fine for
    #     this non-tool-calling path. (agentic_graph.py's tool-calling path
    #     needs the Responses API instead -- bind_tools() on Chat Completions
    #     400s on this model with reasoning_effort active; see that file.)
    #   - 'temperature' and 'top_p' are both REJECTED (400: "Only the default
    #     (1) value is supported" / "not supported with this model") --
    #     intentionally never set, same constraint as Anthropic.
    #   - Legacy 'max_tokens' request param is REJECTED; the API demands
    #     'max_completion_tokens' instead. This function's own parameter
    #     stays named max_tokens (matching call_anthropic/call_ollama, and
    #     what configs/models.yaml + run_baseline.py's CALL_KWARG_KEYS
    #     already forward) and is translated internally here.
    #   - A reasoning-tier model can spend its entire max_completion_tokens
    #     budget on invisible reasoning and return empty visible content with
    #     finish_reason="length" and no error -- confirmed live (20-token cap
    #     -> content="", reasoning_tokens=20/20). Must be surfaced as an
    #     error, not a silent empty-string success, same invariant as
    #     call_anthropic's "no text block" guard below.
    client = _get_openai_client()

    def _call() -> LLMCallResult:
        start = time.monotonic()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        latency_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        text = choice.message.content
        finish_reason = choice.finish_reason

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        reasoning_tokens = None
        cached_input_tokens = None
        cache_write_tokens = 0
        if usage is not None:
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details is not None:
                reasoning_tokens = completion_details.reasoning_tokens
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None:
                cached_input_tokens = prompt_details.cached_tokens
                cache_write_tokens = prompt_details.cache_write_tokens or 0

        cost_usd = None
        if model in OPENAI_COST_PER_MTOK and input_tokens is not None:
            in_rate, cached_rate, cache_write_rate, out_rate = OPENAI_COST_PER_MTOK[model]
            cached = cached_input_tokens or 0
            # prompt_tokens is inclusive of cached_tokens (OpenAI convention,
            # confirmed against the live usage object) -- don't double-count.
            uncached_input = max(input_tokens - cached, 0)
            cost_usd = (
                uncached_input * in_rate
                + cached * cached_rate
                + cache_write_tokens * cache_write_rate
                + (output_tokens or 0) * out_rate
            ) / 1_000_000

        # Empty/None visible text with no other error must surface as an
        # error, not a silent success -- this is exactly the reasoning-token-
        # starvation failure mode confirmed live above.
        error = None if text else (
            f"OpenAI response contained no visible text (finish_reason={finish_reason}, "
            f"reasoning_tokens={reasoning_tokens})"
        )

        return LLMCallResult(
            raw_text=text if text else None,
            error=error,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            stop_reason=finish_reason,
            reasoning_tokens=reasoning_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    return _with_retry(_call, max_retries, base_delay_s, max_delay_s, _is_openai_retryable)


def _is_ollama_retryable(e: Exception) -> bool:
    if isinstance(e, requests.exceptions.RequestException):
        response = getattr(e, "response", None)
        if response is not None:
            return response.status_code >= 500
        return True
    return False


def call_ollama(
    prompt: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    num_ctx: int = 8192,
    base_url: str = "http://localhost:11434",
    max_retries: int = 5,
    base_delay_s: float = 2.0,
    max_delay_s: float = 60.0,
    **_ignored,
) -> LLMCallResult:

    def _call() -> LLMCallResult:
        start = time.monotonic()
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": max_tokens,
                },
            },
            timeout=900,
        )
        response.raise_for_status()
        latency_ms = (time.monotonic() - start) * 1000

        data = response.json()

        return LLMCallResult(
            raw_text=data.get("response"),
            error=None,
            latency_ms=latency_ms,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            cost_usd=0.0,
            stop_reason=data.get("done_reason"),
        )

    return _with_retry(_call, max_retries, base_delay_s, max_delay_s, _is_ollama_retryable)


_BACKENDS = {
    "anthropic": call_anthropic,
    "ollama": call_ollama,
    "openai": call_openai,
}


def call_llm(backend: str, model: str, prompt: str, **kwargs) -> LLMCallResult:
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[backend](prompt, model, **kwargs)
