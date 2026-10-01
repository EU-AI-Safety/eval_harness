from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import JudgeConfig, ModelConfig, RunConfig


def _load_json_or_toml(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Run spec must be a JSON object: {path}")
        return payload
    if suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError as exc:
            raise ValueError("TOML config support requires Python 3.11+") from exc
        payload = tomllib.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Run spec must be a TOML table: {path}")
        return payload
    raise ValueError(f"Unsupported run spec format for {path}. Use .json or .toml")


def _as_list(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Expected list for '{key}' in run spec")
    return tuple(str(item) for item in value)


def _merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_alias_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"models": {}, "judges": {}}
    payload = _load_json_or_toml(path)
    models = payload.get("models", {})
    judges = payload.get("judges", {})
    if not isinstance(models, dict) or not isinstance(judges, dict):
        raise ValueError("Alias file must contain 'models' and 'judges' tables")
    return {"models": models, "judges": judges}


def resolve_alias(alias_payload: dict[str, Any], alias_name: str, *, alias_kind: str) -> dict[str, Any]:
    registry_key = "models" if alias_kind == "model" else "judges"
    registry = alias_payload.get(registry_key, {})
    resolved = registry.get(alias_name)
    if resolved is None:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(f"Unknown {alias_kind} alias '{alias_name}'. Available: {available}")
    if not isinstance(resolved, dict):
        raise ValueError(f"Alias '{alias_name}' for {alias_kind} must resolve to a mapping")
    return dict(resolved)


def _model_from_payload(payload: dict[str, Any], *, default_name: str | None = None) -> ModelConfig:
    return ModelConfig(
        name=str(payload.get("name") or default_name or payload["model"]),
        backend=str(payload["backend"]),
        provider=payload.get("provider"),
        mode=payload.get("mode"),
        model=str(payload["model"]),
        api_key=payload.get("api_key"),
        base_url=payload.get("base_url"),
        system_prompt=payload.get("system_prompt"),
        temperature=float(payload.get("temperature", 0.0)),
        top_p=float(payload.get("top_p", 1.0)),
        max_tokens=int(payload.get("max_tokens", 2048)),
        timeout_seconds=float(payload.get("timeout_seconds", 60.0)),
        max_retries=int(payload.get("max_retries", 3)),
        retry_base_delay_seconds=float(payload.get("retry_base_delay_seconds", 1.0)),
        retry_max_delay_seconds=float(payload.get("retry_max_delay_seconds", 8.0)),
        retry_jitter_seconds=float(payload.get("retry_jitter_seconds", 0.25)),
    )


def _judge_from_payload(payload: dict[str, Any], *, default_name: str | None = None) -> JudgeConfig:
    prompt_template_path = payload.get("prompt_template_path")
    return JudgeConfig(
        name=str(payload.get("name") or default_name or payload["model"]),
        backend=str(payload["backend"]),
        provider=payload.get("provider"),
        mode=payload.get("mode"),
        model=str(payload["model"]),
        api_key=payload.get("api_key"),
        base_url=payload.get("base_url"),
        system_prompt=payload.get("system_prompt"),
        temperature=float(payload.get("temperature", 0.0)),
        top_p=float(payload.get("top_p", 1.0)),
        max_tokens=int(payload.get("max_tokens", 512)),
        timeout_seconds=float(payload.get("timeout_seconds", 60.0)),
        max_retries=int(payload.get("max_retries", 3)),
        retry_base_delay_seconds=float(payload.get("retry_base_delay_seconds", 1.0)),
        retry_max_delay_seconds=float(payload.get("retry_max_delay_seconds", 8.0)),
        retry_jitter_seconds=float(payload.get("retry_jitter_seconds", 0.25)),
        prompt_template_path=Path(prompt_template_path) if prompt_template_path else None,
    )


def load_run_config(
    config_path: Path,
    *,
    alias_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> RunConfig:
    payload = _load_json_or_toml(config_path)
    cli_overrides = cli_overrides or {}
    alias_payload = _load_alias_payload(alias_path or (Path(payload["alias_path"]) if payload.get("alias_path") else None))

    model_payload = payload.get("model")
    if model_payload is None and payload.get("model_alias"):
        model_payload = resolve_alias(alias_payload, str(payload["model_alias"]), alias_kind="model")
    elif model_payload is not None and payload.get("model_alias"):
        model_payload = _merge_mapping(
            resolve_alias(alias_payload, str(payload["model_alias"]), alias_kind="model"),
            model_payload,
        )
    if model_payload is None:
        raise ValueError(f"Run spec {config_path} must define either 'model' or 'model_alias'")

    judge_payload = payload.get("judge")
    if judge_payload is None and payload.get("judge_alias"):
        judge_payload = resolve_alias(alias_payload, str(payload["judge_alias"]), alias_kind="judge")
    elif judge_payload is not None and payload.get("judge_alias"):
        judge_payload = _merge_mapping(
            resolve_alias(alias_payload, str(payload["judge_alias"]), alias_kind="judge"),
            judge_payload,
        )

    dataset_path = Path(str(cli_overrides.get("dataset_path") or payload.get("dataset_path") or "eval_harness/data/EU_alert_working_copy/test/test.jsonl"))
    output_dir = Path(str(cli_overrides.get("output_dir") or payload.get("output_dir") or "eval"))

    extra_metadata = dict(payload.get("notes", {})) if isinstance(payload.get("notes"), dict) else {}
    if payload.get("notes") is not None and not isinstance(payload.get("notes"), dict):
        extra_metadata["notes"] = str(payload.get("notes"))
    extra_metadata["config_path"] = str(config_path)
    if alias_path is not None:
        extra_metadata["alias_path"] = str(alias_path)
    if payload.get("model_alias"):
        extra_metadata["model_alias"] = str(payload.get("model_alias"))
    if payload.get("judge_alias"):
        extra_metadata["judge_alias"] = str(payload.get("judge_alias"))

    return RunConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        limit=cli_overrides.get("limit", payload.get("limit")),
        concurrency=int(cli_overrides.get("concurrency", payload.get("concurrency", 8))),
        overwrite=bool(cli_overrides.get("overwrite", payload.get("overwrite", False))),
        verified_only=bool(cli_overrides.get("verified_only", payload.get("verified_only", False))),
        language_include=_as_list(payload, "language_include"),
        language_exclude=_as_list(payload, "language_exclude"),
        category_include=_as_list(payload, "category_include"),
        category_exclude=_as_list(payload, "category_exclude"),
        random_sample=cli_overrides.get("random_sample", payload.get("random_sample")),
        sampling_seed=int(cli_overrides.get("sampling_seed", payload.get("sampling_seed", 0))),
        model=_model_from_payload(model_payload, default_name=str(payload.get("model_alias") or model_payload.get("name", "model"))),
        judge=(
            _judge_from_payload(judge_payload, default_name=str(payload.get("judge_alias") or judge_payload.get("name", "judge")))
            if judge_payload is not None
            else None
        ),
        run_name=str(payload.get("run_name")) if payload.get("run_name") else None,
        allow_row_errors=bool(cli_overrides.get("allow_row_errors", payload.get("allow_row_errors", False))),
        extra_metadata=extra_metadata,
        version=str(payload.get("version", "r1")),
    )


def config_fingerprint(config_path: Path, *, alias_path: Path | None = None) -> str:
    digest = hashlib.sha1()
    digest.update(config_path.read_bytes())
    if alias_path is not None and alias_path.exists():
        digest.update(alias_path.read_bytes())
    return digest.hexdigest()