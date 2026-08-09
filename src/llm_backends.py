import os
import random
import time
from dataclasses import dataclass, field

import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

# $ per million tokens: (input, output)
ANTHROPIC_COST_PER_MTOK = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


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

        return LLMCallResult(
            raw_text=text,
            error=None,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            stop_reason=response.stop_reason,
        )

    return _with_retry(_call, max_retries, base_delay_s, max_delay_s, _is_anthropic_retryable)


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
}


def call_llm(backend: str, model: str, prompt: str, **kwargs) -> LLMCallResult:
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[backend](prompt, model, **kwargs)
