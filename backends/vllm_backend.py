from __future__ import annotations

from threading import Lock
import time
from typing import Optional

from .api_backend import APIBackend
from .base import BackendResult, InferenceBackend


class VLLMBackend(InferenceBackend):
    def __init__(self, *, mode: str = "openai_compatible", **kwargs) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        self._local_llm = None
        self._local_llm_lock = Lock()
        self._openai_delegate: APIBackend | None = None

    def _get_local_llm(self):
        if self._local_llm is not None:
            return self._local_llm
        with self._local_llm_lock:
            if self._local_llm is not None:
                return self._local_llm
            try:
                from vllm import LLM
            except ImportError as exc:
                raise RuntimeError("vllm package is required for local vLLM mode") from exc
            self._local_llm = LLM(model=self.model)
            return self._local_llm

    def _get_openai_delegate(self) -> APIBackend:
        if self._openai_delegate is None:
            self._openai_delegate = APIBackend(
                provider="openai_compatible",
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout_seconds=self.timeout_seconds,
                max_retries=0,
                retry_base_delay_seconds=self.retry_base_delay_seconds,
                retry_max_delay_seconds=self.retry_max_delay_seconds,
                retry_jitter_seconds=self.retry_jitter_seconds,
            )
        return self._openai_delegate

    def _generate_once(self, prompt: str, system_prompt: Optional[str] = None) -> BackendResult:
        if self.mode == "openai_compatible":
            return self._get_openai_delegate().generate(prompt, system_prompt=system_prompt)

        if self.mode == "local":
            try:
                from vllm import SamplingParams
            except ImportError as exc:
                raise RuntimeError("vllm package is required for local vLLM mode") from exc
            started_at = time.time()
            llm = self._get_local_llm()
            full_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
            outputs = llm.generate(
                [full_prompt],
                sampling_params=SamplingParams(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                ),
            )
            text = outputs[0].outputs[0].text
            if not text.strip():
                raise ValueError("Empty completion returned by local vLLM backend")
            return BackendResult(
                text=text,
                metadata={
                    "provider": "vllm_local",
                    "latency_seconds": round(time.time() - started_at, 4),
                },
            )

        raise ValueError(f"Unsupported vLLM mode: {self.mode}")
