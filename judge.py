from __future__ import annotations

import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import time

from .backends import create_backend
from .config import RunConfig
from .dataset import dataset_file_sha1, summarize_dataset
from .io import append_jsonl, compact_jsonl_by_key, iter_jsonl, load_jsonl_index, rewrite_jsonl, write_json
from .run_specs import config_fingerprint
from .types import JudgeRecord


DEFAULT_JUDGE_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "judge_refusal.txt"

_LABEL_LINE_RE = re.compile(
    r"^\s*label\s*:\s*[<\(\[\"'`]*\s*(refused|complied)\s*[>\)\]\"'`]*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_EXPLANATION_LINE_RE = re.compile(r"^\s*explanation\s*:\s*", re.IGNORECASE | re.MULTILINE)


def parse_judge_output(raw_output: str) -> tuple[str, str]:
    """Parse structured judge output into (label, explanation).

    Preferred format:
      label: refused|complied
      explanation: ...

    Also accepts a legacy bare label (or text containing only one label word).
    """
    text = raw_output.strip()
    if not text:
        raise ValueError("Could not parse empty judge output")

    label_match = _LABEL_LINE_RE.search(text)
    explanation_match = _EXPLANATION_LINE_RE.search(text)
    parsed_explanation = ""
    if explanation_match:
        parsed_explanation = text[explanation_match.end() :].strip()

    if label_match:
        label = label_match.group(1).lower()
        after_label = text[label_match.end() :].strip()
        explanation = parsed_explanation or after_label
        return label, explanation

    normalized = text.lower()
    if normalized in {"refused", "complied"}:
        return normalized, parsed_explanation
    if "refused" in normalized and "complied" not in normalized:
        return "refused", parsed_explanation
    if "complied" in normalized and "refused" not in normalized:
        return "complied", parsed_explanation
    raise ValueError(f"Could not parse judge output: {raw_output}")


def normalize_judge_label(raw_output: str) -> str:
    label, _explanation = parse_judge_output(raw_output)
    return label


def _successful_judged_index(path: Path) -> dict[str, dict]:
    index = load_jsonl_index(path, "row_id")
    return {row_id: row for row_id, row in index.items() if row.get("parsing_status") == "success"}


def _judge_template(config: RunConfig) -> str:
    if config.judge and config.judge.prompt_template_path:
        return config.judge.prompt_template_path.read_text(encoding="utf-8")
    return DEFAULT_JUDGE_TEMPLATE_PATH.read_text(encoding="utf-8")


def _judged_path(config: RunConfig) -> Path:
    if config.model is None or config.judge is None:
        raise ValueError("Both model and judge configs are required")
    return config.output_dir / "judged" / f"{config.model.name}__{config.judge.name}.jsonl"


def _manifest_path(config: RunConfig) -> Path:
    return config.output_dir / "manifests" / f"{config.model.name}__{config.judge.name}.judging.json"


def _status_path(config: RunConfig) -> Path:
    return config.output_dir / "manifests" / f"{config.model.name}__{config.judge.name}.judging.status.json"


def plan_judging(config: RunConfig, responses_path: Path | None = None) -> dict:
    if config.judge is None or config.model is None:
        raise ValueError("RunConfig.model and RunConfig.judge must be provided for judging")
    source_path = responses_path or config.output_dir / "responses" / f"{config.resolved_run_name}.jsonl"
    output_path = _judged_path(config)
    completed = {} if config.overwrite else _successful_judged_index(output_path)
    rows = list(iter_jsonl(source_path)) if source_path.exists() else []
    # Deduplicate responses by row_id (last write wins) before planning.
    response_index = {row["row_id"]: row for row in rows}
    pending = [row for row_id, row in response_index.items() if row_id not in completed]
    return {
        "task": "judging_plan",
        "version": config.version,
        "run_name": f"{config.model.name}__{config.judge.name}",
        "dataset": summarize_dataset(config),
        "fingerprints": {
            "config_fingerprint": _config_fingerprint(config),
        },
        "responses_path": str(source_path),
        "responses_exist": source_path.exists(),
        "response_count_total": len(response_index),
        "response_count_pending": len(pending),
        "response_count_completed_existing": len(completed),
        "output_path": str(output_path),
        "overwrite": config.overwrite,
        "judge": {
            "name": config.judge.name,
            "backend": config.judge.backend,
            "provider": config.judge.provider,
            "mode": config.judge.mode,
            "model": config.judge.model,
            "template_path": str(config.judge.prompt_template_path)
            if config.judge.prompt_template_path
            else None,
        },
    }


def run_judging(config: RunConfig, responses_path: Path | None = None) -> Path:
    if config.judge is None or config.model is None:
        raise ValueError("RunConfig.model and RunConfig.judge must be provided for judging")
    source_path = responses_path or config.output_dir / "responses" / f"{config.resolved_run_name}.jsonl"
    output_path = _judged_path(config)
    completed = {} if config.overwrite else _successful_judged_index(output_path)
    template = _judge_template(config)
    judge_backend = create_backend(config.judge)

    response_index = {row["row_id"]: row for row in iter_jsonl(source_path)}
    rows = [row for row_id, row in response_index.items() if row_id not in completed]
    generation_failures = sum(1 for row in rows if row.get("status") != "success")
    write_json(
        _manifest_path(config),
        {
            "task": "judging",
            "version": config.version,
            "run_name": f"{config.model.name}__{config.judge.name}",
            "dataset_path": str(config.dataset_path),
            "dataset_file_sha1": dataset_file_sha1(config.dataset_path),
            "config_fingerprint": _config_fingerprint(config),
            "responses_path": str(source_path),
            "response_count_pending": len(rows),
            "response_count_completed_existing": len(completed),
            "response_count_generation_failures": generation_failures,
            "allow_row_errors": config.allow_row_errors,
            "judge": {
                "name": config.judge.name,
                "backend": config.judge.backend,
                "model": config.judge.model,
                "template_path": str(config.judge.prompt_template_path)
                if config.judge.prompt_template_path
                else None,
            },
            "environment": {"hostname": socket.gethostname()},
        },
    )
    write_json(_status_path(config), {"status": "running", "pending": len(rows)})
    if config.overwrite and output_path.exists():
        output_path.unlink()
    elif completed:
        # Drop prior non-success rows so retries replace them cleanly.
        rewrite_jsonl(output_path, completed.values())

    def process_row(row: dict) -> JudgeRecord:
        started_at = time()
        if row.get("status") != "success":
            return JudgeRecord(
                row_id=row["row_id"],
                target=row["target"],
                judge_label="parse_error",
                expected_behavior="refused" if row["target"] == "should_refuse" else "complied",
                response=row.get("response", ""),
                prompt=row["prompt"],
                model_name=row["model_name"],
                judge_name=config.judge.name,
                language=row["language"],
                eu_ai_act_category=row["eu_ai_act_category"],
                annex_section=row["annex_section"],
                is_machine_translation=row["is_machine_translation"],
                raw_judge_output=row.get("error") or "generation_failed",
                judge_explanation="",
                parsing_status="generation_error",
                metadata={
                    **row.get("metadata", {}),
                    "judge_latency_seconds": round(time() - started_at, 4),
                    "judge_status": "skipped_due_to_generation_error",
                    **config.extra_metadata,
                },
            )
        judge_prompt = template.format(prompt=row["prompt"], response=row["response"])
        try:
            backend_result = judge_backend.generate(judge_prompt)
            judge_label, judge_explanation = parse_judge_output(backend_result.text)
            return JudgeRecord(
                row_id=row["row_id"],
                target=row["target"],
                judge_label=judge_label,
                expected_behavior="refused" if row["target"] == "should_refuse" else "complied",
                response=row["response"],
                prompt=row["prompt"],
                model_name=row["model_name"],
                judge_name=config.judge.name,
                language=row["language"],
                eu_ai_act_category=row["eu_ai_act_category"],
                annex_section=row["annex_section"],
                is_machine_translation=row["is_machine_translation"],
                raw_judge_output=backend_result.text,
                judge_explanation=judge_explanation,
                parsing_status="success",
                metadata={
                    **row.get("metadata", {}),
                    "judge_latency_seconds": round(time() - started_at, 4),
                    "judge_status": "success",
                    **backend_result.metadata,
                    **config.extra_metadata,
                },
            )
        except Exception as exc:
            return JudgeRecord(
                row_id=row["row_id"],
                target=row["target"],
                judge_label="parse_error",
                expected_behavior="refused" if row["target"] == "should_refuse" else "complied",
                response=row["response"],
                prompt=row["prompt"],
                model_name=row["model_name"],
                judge_name=config.judge.name,
                language=row["language"],
                eu_ai_act_category=row["eu_ai_act_category"],
                annex_section=row["annex_section"],
                is_machine_translation=row["is_machine_translation"],
                raw_judge_output=str(exc),
                judge_explanation="",
                parsing_status="error",
                metadata={
                    **row.get("metadata", {}),
                    "judge_latency_seconds": round(time() - started_at, 4),
                    "judge_status": "error",
                    **config.extra_metadata,
                },
            )

    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as executor:
        futures = [executor.submit(process_row, row) for row in rows]
        for future in as_completed(futures):
            append_jsonl(output_path, future.result().to_dict())

    written = compact_jsonl_by_key(output_path, "row_id")
    success_count = sum(1 for row in written.values() if row.get("parsing_status") == "success")
    failure_count = len(written) - success_count
    stage_status = "completed" if failure_count == 0 else "completed_with_errors"
    write_json(
        _status_path(config),
        {
            "status": stage_status,
            "output_path": str(output_path),
            "row_count_written": len(written),
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