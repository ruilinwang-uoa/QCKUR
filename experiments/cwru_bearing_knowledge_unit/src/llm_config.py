"""LLM API configuration for the CWRU bearing experiment.

The CWRU generation campaign uses GLM-4-Flash through Zhipu AI's
OpenAI-compatible endpoint. The Layer-2/Layer-3 judging campaign uses
DeepSeek-V4-Flash through the DeepSeek OpenAI-compatible endpoint.

For compatibility with the existing campaign scripts, the generation provider
key remains ``openai`` even though its default endpoint/model are Zhipu
GLM-4-Flash. Credentials and endpoint/model overrides are read from environment
variables or, optionally, from a local ``.env`` file at the experiment root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

import config as C

ENV_PATH: Path = C.ROOT / ".env"


_PROVIDERS = {
    "openai": "OPENAI",
    "deepseek": "DEEPSEEK",
}

_DEFAULT_BASE_URL = {
    "openai": "https://open.bigmodel.cn/api/paas/v4",
    "deepseek": "https://api.deepseek.com",
}

_DEFAULT_MODEL = {
    "openai": "glm-4-flash",
    "deepseek": "deepseek-v4-flash",
}


def _load_env(path: Path) -> dict[str, str]:
    """Load key-value pairs from an optional local .env file."""
    if not path.exists():
        return {}

    try:
        from dotenv import dotenv_values  # type: ignore

        return {
            key: value
            for key, value in dotenv_values(path).items()
            if value is not None
        }
    except Exception:
        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
        return values


_ENV = _load_env(ENV_PATH)


def _get(key: str, default: str = "") -> str:
    """Prefer a real environment variable, then .env, then the supplied default."""
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
    extra_body: dict | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.base_url) and bool(self.model)

    @property
    def version_string(self) -> str:
        return self.model


def get_provider(name: str) -> ProviderConfig:
    """Return the configuration for a supported provider key."""
    if name not in _PROVIDERS:
        raise KeyError(
            f"unknown provider '{name}'; choose from {list(_PROVIDERS)}"
        )

    prefix = _PROVIDERS[name]
    return ProviderConfig(
        name=name,
        api_key=_get(f"{prefix}_API_KEY"),
        base_url=_get(f"{prefix}_BASE_URL", _DEFAULT_BASE_URL[name]),
        model=_get(f"{prefix}_MODEL", _DEFAULT_MODEL[name]),
        max_tokens=int(_get("LLM_MAX_TOKENS", "512")),
        temperature=float(_get("LLM_TEMPERATURE", "0.3")),
        timeout=int(_get("LLM_REQUEST_TIMEOUT", "60")),
        max_retries=int(_get("LLM_MAX_RETRIES", "3")),
        extra_body=None,
    )


def available_providers() -> list[str]:
    """Return provider keys with configured API credentials."""
    return [
        name
        for name in _PROVIDERS
        if get_provider(name).is_configured
    ]


def build_client(name: str) -> OpenAI:
    """Build an OpenAI-compatible client for the requested provider."""
    cfg = get_provider(name)
    if not cfg.is_configured:
        prefix = _PROVIDERS[name]
        raise RuntimeError(
            f"provider '{name}' is not configured: set "
            f"{prefix}_API_KEY (and override {prefix}_BASE_URL / "
            f"{prefix}_MODEL if required)"
        )

    return OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )


def status() -> dict:
    """Return a configuration summary without exposing API-key values."""
    return {
        "env_file": str(ENV_PATH),
        "env_file_exists": ENV_PATH.exists(),
        "providers": {
            name: {
                "configured": get_provider(name).is_configured,
                "base_url": get_provider(name).base_url,
                "model": get_provider(name).model,
                "api_key_set": bool(get_provider(name).api_key),
            }
            for name in _PROVIDERS
        },
        "defaults": {
            "max_tokens": get_provider("openai").max_tokens,
            "temperature": get_provider("openai").temperature,
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(status(), indent=2, ensure_ascii=False))
