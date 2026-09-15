"""LLM API configuration loaded from the external ``.env`` file.

Centralizes every provider setting (base URL, API key, model + version date) and
the paper's unified generation defaults (max_tokens=512, temperature=0.3). The
.env file lives at the experiment root and is git-ignored; copy .env.example to
.env and fill in your values.

Both OpenAI and DeepSeek speak the OpenAI-compatible Chat Completions API, so a
single :class:`openai.OpenAI` client works for either, parameterised by base_url.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

import config as C

ENV_PATH: Path = C.ROOT / ".env"

# provider -> env-var prefix
_PROVIDERS = {
    "openai": "OPENAI",
    "deepseek": "DEEPSEEK",
    # third judge family (Zhipu GLM-5.3) for re-judging the DeepSeek-generated
    # cross-LLM reports, removing the self-judgment caveat of Table (cross-LLM)
    "glm53": "GLM53",
    # fourth judge family (Moonshot Kimi K2.6, via SiliconFlow) for re-judging the PRIMARY
    # (GLM-4-Flash-generated) reports: a family disjoint from both the GLM
    # generator and the DeepSeek primary judge (paper Section 5.3, judge
    # battery; OpenAI-compatible coding endpoint)
    "kimi": "KIMI",
}

_DEFAULT_BASE_URL = {
    "openai": "https://open.bigmodel.cn/api/paas/v4",
    "glm53": "https://open.bigmodel.cn/api/paas/v4",
    "deepseek": "https://api.deepseek.com",
    "kimi": "https://api.siliconflow.cn/v1",
}
_DEFAULT_MODEL = {
    "openai": "glm-4-flash",
    "glm53": "glm-5.3",
    "deepseek": "deepseek-v4-flash",
    "kimi": "Pro/moonshotai/Kimi-K2.6",
}

# Provider-specific request params. Kimi-K2.6 defaults to thinking mode, which
# emits ~1-2k reasoning tokens per judging call (slow and it can truncate the
# strict JSON against the token budget); SiliconFlow exposes an explicit
# switch. Non-thinking mode matches the non-reasoning primary generation
# backend and lets the judging passes run at the primary budget
# (max_tokens 1500), removing judge-budget as a covariate.
_DEFAULT_EXTRA_BODY = {
    "kimi": {"enable_thinking": False},
}


def _load_env(path: Path) -> dict[str, str]:
    """Minimal .env parser (no hard dependency on python-dotenv)."""
    if not path.exists():
        return {}
    try:
        from dotenv import dotenv_values  # type: ignore
        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    except Exception:
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("\"'")
        return out


_ENV = _load_env(ENV_PATH)


def _get(key: str, default: str = "") -> str:
    # prefer real environment variable, fall back to .env
    return os.environ.get(key, _ENV.get(key, default)) or default


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    max_tokens: int
    temperature: float
    timeout: int
    max_retries: int
    extra_body: dict | None = None  # provider-specific params (e.g. enable_thinking)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.base_url) and bool(self.model)

    @property
    def version_string(self) -> str:
        """Concrete model string for the paper (report in §4.4)."""
        return self.model


def get_provider(name: str) -> ProviderConfig:
    if name not in _PROVIDERS:
        raise KeyError(f"unknown provider '{name}'; choose from {list(_PROVIDERS)}")
    p = _PROVIDERS[name]
    return ProviderConfig(
        name=name,
        api_key=_get(f"{p}_API_KEY"),
        base_url=_get(f"{p}_BASE_URL", _DEFAULT_BASE_URL[name]),
        model=_get(f"{p}_MODEL", _DEFAULT_MODEL[name]),
        max_tokens=int(_get("LLM_MAX_TOKENS", "512")),
        temperature=float(_get("LLM_TEMPERATURE", "0.3")),
        timeout=int(_get("LLM_REQUEST_TIMEOUT", "60")),
        max_retries=int(_get("LLM_MAX_RETRIES", "3")),
        extra_body=_DEFAULT_EXTRA_BODY.get(name),
    )


def available_providers() -> list[str]:
    """Providers with a non-empty API key in .env / environment."""
    return [n for n in _PROVIDERS if get_provider(n).is_configured]


def build_client(name: str) -> OpenAI:
    cfg = get_provider(name)
    if not cfg.is_configured:
        raise RuntimeError(
            f"provider '{name}' is not configured: set {name.upper()}_API_KEY and "
            f"{name.upper()}_BASE_URL in {ENV_PATH}")
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url,
                  timeout=cfg.timeout, max_retries=cfg.max_retries)


def embedding_model() -> str | None:
    m = _get("EMBEDDING_MODEL")
    return m or None


def status() -> dict:
    """Human-readable configuration summary (keys masked)."""
    return {
        "env_file": str(ENV_PATH),
        "env_file_exists": ENV_PATH.exists(),
        "providers": {
            n: {"configured": get_provider(n).is_configured,
                "base_url": get_provider(n).base_url,
                "version": get_provider(n).version_string,
                "api_key_set": "***" if get_provider(n).api_key else "(empty)"}
            for n in _PROVIDERS
        },
        "defaults": {"max_tokens": get_provider("openai").max_tokens,
                     "temperature": get_provider("openai").temperature},
        "embedding_model": embedding_model() or "(disabled)",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2, ensure_ascii=False))
