"""Connectivity smoke test for configured LLM providers.

Run after filling in `.env`. Pings each configured provider with a 1-token
completion and reports latency. Skips providers whose API key is empty. Use this
to verify URLs/keys before launching the full B3-B8 campaign.
"""
from __future__ import annotations

import time

import llm_config


def smoke(provider: str) -> dict:
    cfg = llm_config.get_provider(provider)
    if not cfg.is_configured:
        return {"provider": provider, "status": "skipped",
                "reason": f"{provider.upper()}_API_KEY empty in .env"}
    try:
        client = llm_config.build_client(provider)
        t0 = time.time()
        resp = client.chat.completions.create(
            model=cfg.model,
            max_tokens=1,
            temperature=cfg.temperature,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        )
        dt = time.time() - t0
        return {"provider": provider, "status": "ok", "model": cfg.version_string,
                "base_url": cfg.base_url, "latency_s": round(dt, 2),
                "reply": resp.choices[0].message.content}
    except Exception as e:  # noqa: BLE001
        return {"provider": provider, "status": "error",
                "model": cfg.version_string, "base_url": cfg.base_url,
                "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    print("LLM config status:")
    for k, v in llm_config.status().items():
        print(f"  {k}: {v}")
    print("\nSmoke test:")
    for p in ["openai", "deepseek"]:
        r = smoke(p)
        print(f"  - {r}")
    avail = llm_config.available_providers()
    if not avail:
        print("\nNo provider configured. Fill OPENAI_API_KEY / DEEPSEEK_API_KEY in .env.")


if __name__ == "__main__":
    main()
