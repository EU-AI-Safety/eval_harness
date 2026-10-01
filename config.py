from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


SUPPORTED_BACKENDS = {"api", "vllm", "openai", "anthropic", "together", "openai_compatible", "vllm_openai"}
SUPPORTED_API_PROVIDERS = {"openai", "anthropic", "together", "openrouter", "openai_compatible"}
SUPPORTED_VLLM_MODES = {"local", "server", "openai_compatible"}

API_KEY_REQUIRED_BACKENDS = {"openai", "anthropic", "together", "openai_compatible", "vllm_openai", "api"}
BASE_URL_REQUIRED_BACKENDS = {"openai_compatible", "vllm_openai"}


def _expand_env(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return os.path.expandvars(value)


def _validate_backend(name: str, backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"Unsupported backend for {name}: {backend}. Supported: {supported}")
    return normalized


def _normalize_optional_token(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _resolve_backend_identity(backend: str, provider: Optional[str], mode: Optional[str]) -> tuple[str, Optional[str], Optional[str]]:
    backend_normalized = _validate_backend("model.backend", backend)
    provider_normalized = _normalize_optional_token(provider)
    mode_normalized = _normalize_optional_token(mode)

    if backend_normalized == "api":
        if provider_normalized is None:
            raise ValueError("provider is required when backend=api")
        if provider_normalized not in SUPPORTED_API_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_API_PROVIDERS))
            raise ValueError(f"Unsupported api provider: {provider}. Supported: {supported}")
        if mode_normalized is not None:
            raise ValueError("mode must not be set when backend=api")
        resolved_provider = "openai_compatible" if provider_normalized == "openrouter" else provider_normalized
        return backend_normalized, resolved_provider, None

    if backend_normalized == "vllm":
        if provider_normalized is not None:
            raise ValueError("provider must not be set when backend=vllm")
        if mode_normalized is None:
            mode_normalized = "server"
        if mode_normalized not in SUPPORTED_VLLM_MODES:
            supported = ", ".join(sorted(SUPPORTED_VLLM_MODES))
            raise ValueError(f"Unsupported vllm mode: {mode}. Supported: {supported}")
        return backend_normalized, None, mode_normalized

    return backend_normalized, provider_normalized, mode_normalized


def _validate_non_empty(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(slots=True)
class ModelConfig:
    name: str
    backend: str
    model: str
    provider: Optional[str] = None
    mode: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 8.0
    retry_jitter_seconds: float = 0.25

    def __post_init__(self) -> None:
        self.name = _validate_non_empty("model.name", self.name)
        self.backend, self.provider, self.mode = _resolve_backend_identity(
            self.backend,
            self.provider,
            self.mode,
        )
        self.model = _validate_non_empty("model.model", self.model)
        self.api_key = _expand_env(self.api_key)
        self.base_url = _expand_env(self.base_url)
        self.system_prompt = _expand_env(self.system_prompt)
        if self.max_tokens <= 0:
            raise ValueError("model.max_tokens must be > 0")
        if self.temperature < 0:
            raise ValueError("model.temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("model.top_p must be within (0, 1]")
        if self.timeout_seconds <= 0:
            raise ValueError("model.timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise ValueError("model.max_retries must be >= 0")
        if self.retry_base_delay_seconds < 0:
            raise ValueError("model.retry_base_delay_seconds must be >= 0")
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError(
                "model.retry_max_delay_seconds must be >= model.retry_base_delay_seconds"
            )
        if self.retry_jitter_seconds < 0:
            raise ValueError("model.retry_jitter_seconds must be >= 0")
        if self.requires_api_key and not self.api_key:
            raise ValueError(f"model.api_key is required for backend: {self.backend}")
        if self.requires_base_url and not self.base_url:
            raise ValueError(f"model.base_url is required for backend: {self.resolved_backend_label}")

    @property
    def resolved_backend_label(self) -> str:
        if self.backend == "api":
            return self.provider or "api"
        if self.backend == "vllm":
            return "vllm_openai" if self.mode in {"server", "openai_compatible"} else "vllm"
        return self.backend

    @property
    def requires_api_key(self) -> bool:
        if self.backend == "api":
            return True
        return self.backend in API_KEY_REQUIRED_BACKENDS

    @property
    def requires_base_url(self) -> bool:
        if self.backend == "api":
            return self.provider in {"openrouter", "openai_compatible"}
        if self.backend == "vllm":
            return self.mode in {"server", "openai_compatible"}
        return self.backend in BASE_URL_REQUIRED_BACKENDS


@dataclass(slots=True)
class JudgeConfig(ModelConfig):
    prompt_template_path: Optional[Path] = None
    allowed_labels: tuple[str, str] = ("refused", "complied")

    def __post_init__(self) -> None:
        # Explicit call: super() breaks with slotted dataclass inheritance.
        ModelConfig.__post_init__(self)
        if self.prompt_template_path is not None and not self.prompt_template_path.exists():
            raise ValueError(
                f"judge.prompt_template_path does not exist: {self.prompt_template_path}"
            )
        if len(self.allowed_labels) != 2:
            raise ValueError("judge.allowed_labels must contain exactly two labels")


@dataclass(slots=True)
class RunConfig:
    dataset_path: Path
    output_dir: Path = Path("eval")
    limit: Optional[int] = None
    concurrency: int = 8
    overwrite: bool = False
    verified_only: bool = False
    language_include: tuple[str, ...] = ()
    language_exclude: tuple[str, ...] = ()
    category_include: tuple[str, ...] = ()
    category_exclude: tuple[str, ...] = ()
    random_sample: Optional[int] = None
    sampling_seed: int = 0
    model: Optional[ModelConfig] = None
    judge: Optional[JudgeConfig] = None
    run_name: Optional[str] = None
    # Completeness contract: any row-level error makes the stage incomplete unless opted out.
    allow_row_errors: bool = False
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "r1"

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        if not self.dataset_path.exists():
            raise ValueError(f"dataset_path does not exist: {self.dataset_path}")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be > 0 when provided")
        if self.random_sample is not None and self.random_sample <= 0:
            raise ValueError("random_sample must be > 0 when provided")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        if self.limit is not None and self.random_sample is not None:
            raise ValueError("limit and random_sample are mutually exclusive; choose one slicing mode")
        self.run_name = self.run_name.strip() if self.run_name else None
        self.language_include = tuple(x.strip().lower() for x in self.language_include if x.strip())
        self.language_exclude = tuple(x.strip().lower() for x in self.language_exclude if x.strip())
        self.category_include = tuple(x.strip() for x in self.category_include if x.strip())
        self.category_exclude = tuple(x.strip() for x in self.category_exclude if x.strip())
        overlap_languages = sorted(set(self.language_include) & set(self.language_exclude))
        if overlap_languages:
            raise ValueError(f"language_include and language_exclude overlap: {overlap_languages}")
        overlap_categories = sorted(set(self.category_include) & set(self.category_exclude))
        if overlap_categories:
            raise ValueError(f"category_include and category_exclude overlap: {overlap_categories}")

    @property
    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        if self.model:
            return self.model.name
        return "eval_run"