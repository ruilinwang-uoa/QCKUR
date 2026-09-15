"""Thin chat-completion wrapper with token / latency / cost / IFR logging.

Every LLM-based system (B3-B8) routes through :func:`chat` so that the unified
runner can record comparable resource metrics. Cost is tracked per 1K tokens via
a small price table (editable here); when a model is unknown the cost is 0 but
tokens are still logged.
"""
from __future__ import annotations

import time

import llm_config

try:
    from openai import RateLimitError, APIError, APITimeoutError
except Exception:  # noqa: BLE001  (older openai versions)
    RateLimitError = APIError = APITimeoutError = None

# global override for max_tokens (set by the runner; reasoning models need a
# larger budget so chain-of-thought does not consume the whole completion)
_MAX_TOKENS_OVERRIDE: int | None = None


def set_max_tokens(n: int | None) -> None:
    global _MAX_TOKENS_OVERRIDE
    _MAX_TOKENS_OVERRIDE = n

# price per 1K tokens (USD); add entries for your provider's models if desired.
# Unknown models -> cost 0.0 (tokens still recorded).
COST_PER_1K: dict[str, tuple[float, float]] = {
    # "model-name": (input_per_1k, output_per_1k),
}


class LLMError(Exception):
    pass


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = COST_PER_1K.get(model)
    if not p:
        return 0.0
    return round(prompt_tokens / 1000 * p[0] + completion_tokens / 1000 * p[1], 6)


def chat(provider: str, messages: list[dict], *,
         max_tokens: int | None = None, temperature: float | None = None,
         tools: list[dict] | None = None, tool_choice: str | dict | None = None,
         response_format: dict | None = None, seed: int | None = None) -> dict:
    """Call the configured provider and return a metrics dict.

    Returns: {ok, text, tool_calls, prompt_tokens, completion_tokens,
              latency_s, cost_usd, model, error}
    """
    cfg = llm_config.get_provider(provider)
    client = llm_config.build_client(provider)
    if max_tokens is None:
        max_tokens = _MAX_TOKENS_OVERRIDE if _MAX_TOKENS_OVERRIDE else cfg.max_tokens
    kwargs = dict(
        model=cfg.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature if temperature is not None else cfg.temperature,
    )
    if seed is not None:
        kwargs["seed"] = seed
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
    if response_format:
        kwargs["response_format"] = response_format
    if cfg.extra_body:
        kwargs["extra_body"] = dict(cfg.extra_body)

    t0 = time.time()
    # backoff retry for free-tier / transient rate limits (429) and timeouts
    max_rl_retries = 4
    last_err = None
    for attempt in range(max_rl_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            is_rate = RateLimitError is not None and isinstance(e, RateLimitError)
            is_transient = any(p is not None and isinstance(e, p)
                               for p in (APIError, APITimeoutError))
            if attempt < max_rl_retries and (is_rate or is_transient):
                time.sleep(2.0 * (2 ** attempt))   # 2, 4, 8, 16 s
                continue
            return {"ok": False, "text": "", "tool_calls": [],
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "latency_s": round(time.time() - t0, 3), "cost_usd": 0.0,
                    "model": cfg.model, "error": f"{type(e).__name__}: {e}"}
    else:
        return {"ok": False, "text": "", "tool_calls": [],
                "prompt_tokens": 0, "completion_tokens": 0,
                "latency_s": round(time.time() - t0, 3), "cost_usd": 0.0,
                "model": cfg.model, "error": f"retries exhausted: {last_err}"}

    latency = round(time.time() - t0, 3)
    choice = resp.choices[0].message
    text = (choice.content or "").strip()
    tool_calls = []
    if getattr(choice, "tool_calls", None):
        for tc in choice.tool_calls:
            tool_calls.append({"id": tc.id, "name": tc.function.name,
                               "args": tc.function.arguments})
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    return {"ok": True, "text": text, "tool_calls": tool_calls,
            "prompt_tokens": pt, "completion_tokens": ct,
            "latency_s": latency, "cost_usd": estimate_cost(cfg.model, pt, ct),
            "model": cfg.model, "error": None}


def is_ifr(result: dict) -> bool:
    """Intermittent failure: no usable answer returned."""
    if not result.get("ok"):
        return True
    if not result.get("text"):
        return True
    # obvious refusals
    low = result["text"].lower()
    if low.startswith(("i'm sorry", "i cannot", "as an ai")):
        return True
    return False
