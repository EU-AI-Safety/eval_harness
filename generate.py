from __future__ import annotations

import hashlib
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import time

from .backends import create_backend
from .config import RunConfig
from .dataset import dataset_file_sha1, dataset_slice_sha1, load_dataset_rows, summarize_dataset
from .io import append_jsonl, compact_jsonl_by_key, load_jsonl_index, rewrite_jsonl, write_json
from .run_specs import config_fingerprint
from .types import GenerationRecord


def _response_path(config: RunConfig) -> Path:
    return config.output_dir / "responses" / f"{config.resolved_run_name}.jsonl"


def _manifest_path(config: RunConfig) -> Path:
    return config.output_dir / "manifests" / f"{config.resolved_run_name}.generation.json"


def _status_path(config: RunConfig) -> Path:
    return config.output_dir / "manifests" / f"{config.resolved_run_name}.generation.status.json"


def _successful_response_index(path: Path) -> dict[str, dict]:
    index = load_jsonl_index(path, "row_id")
    return {row_id: row for row_id, row in index.items() if row.get("status") == "success"}


def plan_generation(config: RunConfig) -> dict:
    rows = load_dataset_rows(
        config.dataset_path,
        verified_only=config.verified_only,
        limit=config.limit,
        language_include=config.language_include,
        language_exclude=config.language_exclude,
        category_include=config.category_include,
        category_exclude=config.category_exclude,
        random_sample=config.random_sample,
        sampling_seed=config.sampling_seed,
    )
    output_path = _response_path(config)
    completed = {} if config.overwrite else _successful_response_index(output_path)
    summary = summarize_dataset(config)
    return {
        "task": "generation_plan",
        "version": config.version,
        "run_name": config.resolved_run_name,
        "dataset": summary,
        "fingerprints": {
            "dataset_file_sha1": summary["dataset_file_sha1"],
            "dataset_slice_sha1": summary["dataset_slice_sha1"],
            "config_fingerprint": _config_fingerprint(config),
        },
        "output_path": str(output_path),
        "row_count_pending": sum(1 for row in rows if row.row_id not in completed),
        "row_count_completed_existing": len(completed),
        "overwrite": config.overwrite,
        "model": {
            "name": config.model.name if config.model else None,
            "backend": config.model.backend if config.model else None,
            "provider": config.model.provider if config.model else None,
            "mode": config.model.mode if config.model else None,
            "model": config.model.model if config.model else None,
            "temperature": config.model.temperature if config.model else None,
            "top_p": config.model.top_p if config.model else None,
            "max_tokens": config.model.max_tokens if config.model else None,
        },
    }


def run_generation(config: RunConfig) -> Path:
    if config.model is None:
        raise ValueError("RunConfig.model must be provided for generation")

    rows = load_dataset_rows(
        config.dataset_path,
        verified_only=config.verified_only,
        limit=config.limit,
        language_include=config.language_include,
        language_exclude=config.language_exclude,
        category_include=config.category_include,
        category_exclude=config.category_exclude,
        random_sample=config.random_sample,
        sampling_seed=config.sampling_seed,
    )
    output_path = _response_path(config)
    completed = {} if config.overwrite else _successful_response_index(output_path)
    backend = create_backend(config.model)
    pending = [row for row in rows if row.row_id not in completed]

    manifest = {
        "task": "generation",
        "version": config.version,
        "run_name": config.resolved_run_name,
        "dataset_path": str(config.dataset_path),
        "dataset_file_sha1": dataset_file_sha1(config.dataset_path),
        "dataset_slice_sha1": dataset_slice_sha1(rows),
        "config_fingerprint": _config_fingerprint(config),
        "row_count_total": len(rows),
        "row_count_pending": len(pending),
        "row_count_completed_existing": len(completed),
        "allow_row_errors": config.allow_row_errors,
        "filters": {
            "verified_only": config.verified_only,
            "language_include": list(config.language_include),
            "language_exclude": list(config.language_exclude),
            "category_include": list(config.category_include),
            "category_exclude": list(config.category_exclude),
            "limit": config.limit,
            "random_sample": config.random_sample,
            "sampling_seed": config.sampling_seed,
        },
        "model": {
            "name": config.model.name,
            "backend": config.model.backend,
            "model": config.model.model,
            "temperature": config.model.temperature,
            "top_p": config.model.top_p,
            "max_tokens": config.model.max_tokens,
        },
        "environment": {
            "hostname": socket.gethostname(),
        },
    }
    write_json(_manifest_path(config), manifest)
    write_json(_status_path(config), {"status": "running", "pending": len(pending)})

    def process_row(row):
        started_at = time()
        try:
            backend_result = backend.generate(row.prompt, system_prompt=config.model.system_prompt)
            response_text = backend_result.text.strip()
            if not response_text:
                raise ValueError("Empty completion returned by backend")
            return GenerationRecord(
                row_id=row.row_id,
                prompt=row.prompt,
                target=row.target,
                response=response_text,
                language=row.language,
                eu_ai_act_category=row.eu_ai_act_category,
                annex_section=row.annex_section,
                is_machine_translation=row.is_machine_translation,
                model_name=config.model.name,
                backend=config.model.backend,
                prompt_hash=hashlib.sha1(row.prompt.encode("utf-8")).hexdigest(),
                metadata={
                    "source_index": row.source_index,
                    "latency_seconds": round(time() - started_at, 4),
                    "generation_status": "success",
                    **backend_result.metadata,
                    **config.extra_metadata,
                },
            )
        except Exception as exc:
            return GenerationRecord(
                row_id=row.row_id,
                prompt=row.prompt,
                target=row.target,
                response="",
                language=row.language,
                eu_ai_act_category=row.eu_ai_act_category,
                annex_section=row.annex_section,
                is_machine_translation=row.is_machine_translation,
                model_name=config.model.name,
                backend=config.model.backend,
                prompt_hash=hashlib.sha1(row.prompt.encode("utf-8")).hexdigest(),
                status="error",
                error=str(exc),
                metadata={
                    "source_index": row.source_index,
                    "latency_seconds": round(time() - started_at, 4),
                    "generation_status": "error",
                    **config.extra_metadata,
                },
            )

    if config.overwrite and output_path.exists():
        output_path.unlink()
    elif completed:
        # Drop prior non-success rows so retries replace them cleanly.
        rewrite_jsonl(output_path, completed.values())

    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as executor:
        futures = [executor.submit(process_row, row) for row in pending]
        for future in as_completed(futures):
            record = future.result()
            append_jsonl(output_path, record.to_dict())

    written_rows = compact_jsonl_by_key(output_path, "row_id")
    failure_count = sum(1 for row in written_rows.values() if row.get("status") != "success")
    success_count = len(written_rows) - failure_count
    stage_status = "completed" if failure_count == 0 else "completed_with_errors"
    write_json(
        _status_path(config),
        {
            "status": stage_status,
            "output_path": str(output_path),
            "row_count_total": len(rows),
            "row_count_pending": len(pending),
            "row_count_written": len(written_rows),
            "success_count": success_count,
            "failure_count": failure_count,
            "allow_row_errors": config.allow_row_errors,
            "complete": failure_count == 0,
        },
    )

    return output_path


def _config_fingerprint(config: RunConfig) -> str | None:
    config_path = config.extra_metadata.get("config_path")
    if not config_path:
        return None
    alias_path = config.extra_metadata.get("alias_path")
    return config_fingerprint(Path(config_path), alias_path=Path(alias_path) if alias_path else None)