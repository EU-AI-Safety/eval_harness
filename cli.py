from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import JudgeConfig, ModelConfig, RunConfig
from .dataset import summarize_dataset
from .generate import plan_generation, run_generation
from .judge import plan_judging, run_judging
from .metrics import score_judged_file
from .run_specs import load_run_config
from .validate import validate_run_artifacts

DEFAULT_DATASET_PATH = "eval_harness/data/EU_alert_working_copy/test/test.jsonl"
DEFAULT_JUDGE_TEMPLATE_PATH = "eval_harness/prompts/judge_refusal.txt"
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def _resolve_default_dataset_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    if path.is_absolute() or path.exists():
        return path

    candidates = [
        REPO_ROOT / path,
        PACKAGE_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _display_dataset_path(dataset_path: Path) -> str:
    for base in (REPO_ROOT, PACKAGE_ROOT):
        try:
            return str(dataset_path.relative_to(base))
        except ValueError:
            continue
    return str(dataset_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run model evaluations, judging, scoring, and validation for EU-Guard."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(subparser: argparse.ArgumentParser, *, include_judge: bool = False) -> None:
        subparser.add_argument(
            "--dataset-path",
            default=DEFAULT_DATASET_PATH,
            help="Path to the dataset file to evaluate.",
        )
        subparser.add_argument("--output-dir", default="eval", help="Folder where outputs will be written.")
        subparser.add_argument("--config", help="Path to a JSON or TOML run config file.")
        subparser.add_argument("--alias-file", help="Path to an alias registry file used by the config.")
        subparser.add_argument("--limit", type=int, help="Run only the first N dataset rows after filtering.")
        subparser.add_argument("--random-sample", type=int, help="Randomly sample N rows instead of using all rows.")
        subparser.add_argument("--sampling-seed", type=int, default=0, help="Random seed for sampling.")
        subparser.add_argument("--concurrency", type=int, default=8, help="Number of requests to run in parallel.")
        subparser.add_argument(
            "--verified-only",
            action="store_true",
            help="Use only rows that are not marked as machine translated.",
        )
        subparser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs instead of resuming.")
        subparser.add_argument(
            "--allow-row-errors",
            action="store_true",
            help=(
                "Allow the stage/run to exit successfully even if some rows failed. "
                "By default any row-level error makes the run incomplete (exit code 1)."
            ),
        )
        subparser.add_argument("--run-name", help="Custom name for this run.")
        subparser.add_argument("--language-include", action="append", default=[], help="Only include these languages. Repeat to add more.")
        subparser.add_argument("--language-exclude", action="append", default=[], help="Exclude these languages. Repeat to add more.")
        subparser.add_argument("--category-include", action="append", default=[], help="Only include these categories. Repeat to add more.")
        subparser.add_argument("--category-exclude", action="append", default=[], help="Exclude these categories. Repeat to add more.")
        subparser.add_argument("--model-name", help="Short name used to label the model in outputs.")
        subparser.add_argument("--backend", help="Backend type, for example 'api' or 'vllm'.")
        subparser.add_argument("--provider", help="API provider name, for example 'openai' or 'anthropic'.")
        subparser.add_argument(
            "--mode",
            help="Backend mode, for example 'local' or 'server'. For --backend vllm, defaults to 'server'.",
        )
        subparser.add_argument("--model", help="Model identifier sent to the backend.")
        subparser.add_argument("--api-key", help="API key for the selected provider.")
        subparser.add_argument("--base-url", help="Base URL for OpenAI-compatible or server backends.")
        subparser.add_argument("--system-prompt", help="Optional system prompt to send with each generation request.")
        subparser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
        subparser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value for generation.")
        subparser.add_argument("--max-tokens", type=int, default=2048, help="Maximum number of output tokens for model generation.")
        subparser.add_argument("--timeout-seconds", type=float, default=60.0, help="Request timeout in seconds.")
        subparser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries after transient failures.")
        subparser.add_argument("--retry-base-delay-seconds", type=float, default=1.0, help="Base delay in seconds before retrying.")
        subparser.add_argument("--retry-max-delay-seconds", type=float, default=8.0, help="Maximum delay in seconds between retries.")
        subparser.add_argument("--retry-jitter-seconds", type=float, default=0.25, help="Random jitter added to retry delays.")
        if include_judge:
            subparser.add_argument("--judge-name", help="Short name used to label the judge in outputs.")
            subparser.add_argument("--judge-backend", help="Judge backend type, for example 'api' or 'vllm'.")
            subparser.add_argument("--judge-provider", help="Judge API provider name.")
            subparser.add_argument(
                "--judge-mode",
                help="Judge backend mode, for example 'local' or 'server'. For --judge-backend vllm, defaults to 'server'.",
            )
            subparser.add_argument("--judge-model", help="Judge model identifier sent to the backend.")
            subparser.add_argument("--judge-api-key", help="API key for the judge provider.")
            subparser.add_argument("--judge-base-url", help="Base URL for an OpenAI-compatible judge backend.")
            subparser.add_argument("--judge-system-prompt", help="Optional system prompt for the judge.")
            subparser.add_argument(
                "--judge-template-path",
                default=DEFAULT_JUDGE_TEMPLATE_PATH,
                help=(
                    "Path to the prompt template used for judging. "
                    f"Defaults to {DEFAULT_JUDGE_TEMPLATE_PATH}."
                ),
            )
            subparser.add_argument("--judge-temperature", type=float, help="Sampling temperature for the judge.")
            subparser.add_argument("--judge-top-p", type=float, help="Top-p sampling value for the judge.")
            subparser.add_argument("--judge-max-tokens", type=int, default=512, help="Maximum number of output tokens for judge generation.")
            subparser.add_argument("--judge-timeout-seconds", type=float, help="Judge request timeout in seconds.")
            subparser.add_argument("--judge-max-retries", type=int, help="Maximum number of judge retries.")
            subparser.add_argument("--judge-retry-base-delay-seconds", type=float, help="Base retry delay for judge requests.")
            subparser.add_argument("--judge-retry-max-delay-seconds", type=float, help="Maximum retry delay for judge requests.")
            subparser.add_argument("--judge-retry-jitter-seconds", type=float, help="Random jitter added to judge retry delays.")

    add_common_args(subparsers.add_parser("generate", help="Generate model responses for a dataset slice.", description="Generate model responses and write them to a JSONL file."))
    add_common_args(subparsers.add_parser("judge", help="Judge existing model responses.", description="Read saved responses, ask a judge model if each one refused or complied, and write judged outputs."), include_judge=True)
    add_common_args(subparsers.add_parser("run-all", help="Run generation, judging, and scoring in one command.", description="Run the full evaluation pipeline: generate responses, judge them, and score the results."), include_judge=True)
    add_common_args(subparsers.add_parser("plan", help="Preview what a run would do without calling any model.", description="Show dataset size, output paths, and pending work before you run generation or judging."), include_judge=True)

    dataset_parser = subparsers.add_parser(
        "dataset-summary",
        help="Show a summary of a dataset slice.",
        description="Preview how many rows will be used after filters and sampling are applied.",
    )
    dataset_parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="Path to the dataset file.")
    dataset_parser.add_argument("--limit", type=int, help="Use only the first N rows after filtering.")
    dataset_parser.add_argument("--random-sample", type=int, help="Randomly sample N rows.")
    dataset_parser.add_argument("--sampling-seed", type=int, default=0, help="Random seed for sampling.")
    dataset_parser.add_argument("--verified-only", action="store_true", help="Use only rows that are not machine translated.")
    dataset_parser.add_argument("--language-include", action="append", default=[], help="Only include these languages. Repeat to add more.")
    dataset_parser.add_argument("--language-exclude", action="append", default=[], help="Exclude these languages. Repeat to add more.")
    dataset_parser.add_argument("--category-include", action="append", default=[], help="Only include these categories. Repeat to add more.")
    dataset_parser.add_argument("--category-exclude", action="append", default=[], help="Exclude these categories. Repeat to add more.")

    score_parser = subparsers.add_parser(
        "score",
        help="Score a judged JSONL file and write reports.",
        description="Compute metrics from a judged file and write JSON and CSV reports.",
    )
    score_parser.add_argument("--judged-path", required=True, help="Path to the judged JSONL file.")
    score_parser.add_argument("--output-dir", default="eval", help="Folder where score outputs will be written.")
    score_parser.add_argument("--run-stem", required=True, help="Base name used for the output report files.")

    validate_parser = subparsers.add_parser(
        "validate-run",
        help="Check that run outputs and manifests are consistent.",
        description="Run lightweight checks on responses, judged outputs, and manifest files.",
    )
    validate_parser.add_argument("--responses-path", help="Path to the generated responses JSONL file.")
    validate_parser.add_argument("--judged-path", help="Path to the judged JSONL file.")
    validate_parser.add_argument("--generation-manifest-path", help="Path to the generation manifest JSON file.")
    validate_parser.add_argument("--judging-manifest-path", help="Path to the judging manifest JSON file.")

    for parser_with_slicing in [
        subparsers.choices["generate"],
        subparsers.choices["judge"],
        subparsers.choices["run-all"],
        dataset_parser,
    ]:
        parser_with_slicing.epilog = (
            "Notes: --limit and --random-sample cannot be used together. "
            "Do not include and exclude the same language or category."
        )

    return parser


def _model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        name=args.model_name,
        backend=args.backend,
        provider=args.provider,
        mode=args.mode,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_base_delay_seconds=args.retry_base_delay_seconds,
        retry_max_delay_seconds=args.retry_max_delay_seconds,
        retry_jitter_seconds=args.retry_jitter_seconds,
    )


def _judge_config_from_args(args: argparse.Namespace) -> JudgeConfig:
    return JudgeConfig(
        name=args.judge_name,
        backend=args.judge_backend,
        provider=args.judge_provider,
        mode=args.judge_mode,
        model=args.judge_model,
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        system_prompt=args.judge_system_prompt,
        max_tokens=args.judge_max_tokens,
        temperature=args.judge_temperature if args.judge_temperature is not None else args.temperature,
        top_p=args.judge_top_p if args.judge_top_p is not None else args.top_p,
        timeout_seconds=args.judge_timeout_seconds or args.timeout_seconds,
        max_retries=args.judge_max_retries if args.judge_max_retries is not None else args.max_retries,
        retry_base_delay_seconds=(
            args.judge_retry_base_delay_seconds
            if args.judge_retry_base_delay_seconds is not None
            else args.retry_base_delay_seconds
        ),
        retry_max_delay_seconds=(
            args.judge_retry_max_delay_seconds
            if args.judge_retry_max_delay_seconds is not None
            else args.retry_max_delay_seconds
        ),
        retry_jitter_seconds=(
            args.judge_retry_jitter_seconds
            if args.judge_retry_jitter_seconds is not None
            else args.retry_jitter_seconds
        ),
        prompt_template_path=Path(args.judge_template_path) if args.judge_template_path else None,
    )


def _run_config_from_args(args: argparse.Namespace, include_judge: bool = False) -> RunConfig:
    if getattr(args, "config", None):
        return load_run_config(
            Path(args.config),
            alias_path=Path(args.alias_file) if getattr(args, "alias_file", None) else None,
            cli_overrides={
                "dataset_path": args.dataset_path,
                "output_dir": args.output_dir,
                "limit": args.limit,
                "random_sample": args.random_sample,
                "sampling_seed": args.sampling_seed,
                "concurrency": args.concurrency,
                "verified_only": args.verified_only,
                "overwrite": args.overwrite,
                "allow_row_errors": getattr(args, "allow_row_errors", False),
            },
        )
    if not args.model_name or not args.backend or not args.model:
        raise ValueError("Without --config, --model-name, --backend, and --model are required")
    if include_judge and (not args.judge_name or not args.judge_backend or not args.judge_model):
        raise ValueError("Without --config, --judge-name, --judge-backend, and --judge-model are required")
    return RunConfig(
        dataset_path=_resolve_default_dataset_path(args.dataset_path),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        concurrency=args.concurrency,
        overwrite=args.overwrite,
        verified_only=args.verified_only,
        language_include=tuple(args.language_include),
        language_exclude=tuple(args.language_exclude),
        category_include=tuple(args.category_include),
        category_exclude=tuple(args.category_exclude),
        random_sample=args.random_sample,
        sampling_seed=args.sampling_seed,
        model=_model_config_from_args(args),
        judge=_judge_config_from_args(args) if include_judge else None,
        run_name=args.run_name,
        allow_row_errors=bool(getattr(args, "allow_row_errors", False)),
    )


def _stage_is_complete(status_path: Path) -> bool:
    if not status_path.exists():
        return False
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if "complete" in payload:
        return bool(payload["complete"])
    return int(payload.get("failure_count", 0)) == 0 and payload.get("status") == "completed"


def _enforce_completeness(config: RunConfig, *status_paths: Path) -> None:
    if config.allow_row_errors:
        return
    incomplete = [str(path) for path in status_paths if not _stage_is_complete(path)]
    if incomplete:
        raise SystemExit(
            "Run incomplete: one or more stages reported row-level errors. "
            "Re-run to retry failed rows, or pass --allow-row-errors to accept partial results. "
            f"Status files: {', '.join(incomplete)}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        config = _run_config_from_args(args)
        run_generation(config)
        _enforce_completeness(
            config,
            config.output_dir / "manifests" / f"{config.resolved_run_name}.generation.status.json",
        )
        return

    if args.command == "judge":
        config = _run_config_from_args(args, include_judge=True)
        run_judging(config)
        _enforce_completeness(
            config,
            config.output_dir
            / "manifests"
            / f"{config.model.name}__{config.judge.name}.judging.status.json",
        )
        return

    if args.command == "run-all":
        config = _run_config_from_args(args, include_judge=True)
        responses_path = run_generation(config)
        judged_path = run_judging(config, responses_path=responses_path)
        score_judged_file(judged_path, config.output_dir, f"{config.model.name}__{config.judge.name}")
        _enforce_completeness(
            config,
            config.output_dir / "manifests" / f"{config.resolved_run_name}.generation.status.json",
            config.output_dir
            / "manifests"
            / f"{config.model.name}__{config.judge.name}.judging.status.json",
        )
        return

    if args.command == "plan":
        config = _run_config_from_args(args, include_judge=True)
        payload = {
            "generation": plan_generation(config),
            "judging": plan_judging(config),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.command == "score":
        score_judged_file(Path(args.judged_path), Path(args.output_dir), args.run_stem)
        return

    if args.command == "dataset-summary":
        resolved_dataset_path = _resolve_default_dataset_path(args.dataset_path)
        config = RunConfig(
            dataset_path=resolved_dataset_path,
            limit=args.limit,
            verified_only=args.verified_only,
            language_include=tuple(args.language_include),
            language_exclude=tuple(args.language_exclude),
            category_include=tuple(args.category_include),
            category_exclude=tuple(args.category_exclude),
            random_sample=args.random_sample,
            sampling_seed=args.sampling_seed,
        )
        payload = summarize_dataset(config)
        payload["dataset_path"] = _display_dataset_path(resolved_dataset_path)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.command == "validate-run":
        payload = validate_run_artifacts(
            responses_path=Path(args.responses_path) if args.responses_path else None,
            judged_path=Path(args.judged_path) if args.judged_path else None,
            generation_manifest_path=(
                Path(args.generation_manifest_path) if args.generation_manifest_path else None
            ),
            judging_manifest_path=(
                Path(args.judging_manifest_path) if args.judging_manifest_path else None
            ),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()