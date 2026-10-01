from __future__ import annotations

import time
from typing import Any, Optional

from .base import BackendResult, InferenceBackend


class APIBackend(InferenceBackend):
    def __init__(self, *, provider: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.provider = provider.lower()
        self._openai_client: Any | None = None
        self._anthropic_client: Any | None = None

    def _load_openai_client(self):
        if self._openai_client is not None:
            return self._openai_client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for API and OpenAI-compatible backends"
            ) from exc
        self._openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._openai_client

    def _load_anthropic_client(self):
        if self._anthropic_client is not None:
            return self._anthropic_client
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package is required for Anthropic backends") from exc
        self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        return self._anthropic_client

    def _generate_once(self, prompt: str, system_prompt: Optional[str] = None) -> BackendResult:
        if self.provider in {"openai", "openai_compatible", "together"}:
            client = self._load_openai_client()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            started_at = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                timeout=self.timeout_seconds,
            )
            message = response.choices[0].message
            text = message.content or ""
            # Some reasoning models put the visible answer in content and chain-of-
            # thought elsewhere; others may leave content empty. Prefer content,
            # then common reasoning fields.
            if not text.strip():
                for attr in ("reasoning_content", "reasoning"):
                    alt = getattr(message, attr, None)
                    if isinstance(alt, str) and alt.strip():
                        text = alt
                        break
            if not text.strip():
                raise ValueError("Empty completion returned by provider")
            return BackendResult(
                text=text,
                metadata={
                    "provider": self.provider,
                    "latency_seconds": round(time.time() - started_at, 4),
                },
            )

        if self.provider == "anthropic":
            client = self._load_anthropic_client()
            started_at = time.time()
            response = client.messages.create(
                model=self.model,
                system=system_prompt or "",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout_seconds,
            )
            parts = []
            for block in response.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            joined = "".join(parts)
            if not joined.strip():
                raise ValueError("Empty completion returned by provider")
            return BackendResult(
                text=joined,
                metadata={
                    "provider": self.provider,
                    "latency_seconds": round(time.time() - started_at, 4),
                },
            )

        raise ValueError(f"Unsupported API provider: {self.provider}")
