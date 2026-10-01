from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eval_harness.config import JudgeConfig, ModelConfig, RunConfig
from eval_harness.dataset import load_dataset_rows, summarize_dataset
from eval_harness.generate import plan_generation, run_generation
from eval_harness.judge import normalize_judge_label, plan_judging, run_judging
from eval_harness.metrics import score_judged_file
from eval_harness.run_specs import config_fingerprint, load_run_config
from eval_harness.validate import validate_run_artifacts


class EvalHarnessSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.dataset_path = self.repo_root / "eval_harness" / "data" / "EU_alert_working_copy" / "test" / "test.jsonl"
        self.judge_template_path = self.repo_root / "eval_harness" / "prompts" / "judge_refusal.txt"

    def test_dataset_summary_and_filters(self) -> None:
        config = RunConfig(dataset_path=self.dataset_path, limit=5, verified_only=True)
        summary = summarize_dataset(config)
        self.assertGreater(summary["n_rows"], 0)
        self.assertEqual(summary["n_machine_translated"], 0)
        self.assertIn("dataset_file_sha1", summary)

        rows = load_dataset_rows(self.dataset_path, limit=5, language_include=("english",))
        self.assertTrue(rows)
        self.assertTrue(all(row.language == "english" for row in rows))

    def test_normalize_judge_label(self) -> None:
        from eval_harness.judge import parse_judge_output

        self.assertEqual(normalize_judge_label("refused"), "refused")
        self.assertEqual(normalize_judge_label("The answer is complied."), "complied")
        label, explanation = parse_judge_output(
            "label: refused\nexplanation: The model declined to help with the request."
        )
        self.assertEqual(label, "refused")
        self.assertIn("declined", explanation.lower())

        # Regression: judge outputs often wrap labels in angle brackets.
        label, explanation = parse_judge_output(
            "label: <refused>\nexplanation: The model refused and cited safety policy."
        )
        self.assertEqual(label, "refused")
        self.assertIn("safety", explanation.lower())

        # Fallback parsing should still preserve explanation text when present.
        label, explanation = parse_judge_output(
            "Result: complied\nexplanation: The model provided direct procedural guidance."
        )
        self.assertEqual(label, "complied")
        self.assertIn("procedural", explanation.lower())
        with self.assertRaises(ValueError):
            normalize_judge_label("unclear")

    def test_end_to_end_with_stub_backends(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            model = ModelConfig(name="stub-model", backend="openai", model="stub", api_key="test-key")
            judge = JudgeConfig(
                name="stub-judge",
                backend="openai",
                model="stub-judge",
                api_key="test-key",
                prompt_template_path=self.judge_template_path,
            )
            config = RunConfig(
                dataset_path=self.dataset_path,
                output_dir=output_dir,
                limit=4,
                model=model,
                judge=judge,
                run_name="smoke",
            )

            from eval_harness.backends.base import BackendResult
            from eval_harness import generate as generate_module
            from eval_harness import judge as judge_module

            class StubModelBackend:
                def generate(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    return BackendResult(text=f"response::{prompt[:20]}", metadata={"stub": "model"})

            class StubJudgeBackend:
                def generate(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    lowered = prompt.lower()
                    label = "refused" if "should refuse" in lowered or "refuse" in lowered else "complied"
                    return BackendResult(
                        text=f"label: {label}\nexplanation: Stub rationale for {label}.",
                        metadata={"stub": "judge"},
                    )

            original_generate_factory = generate_module.create_backend
            original_judge_factory = judge_module.create_backend
            try:
                generate_module.create_backend = lambda cfg: StubModelBackend()
                judge_module.create_backend = (
                    lambda cfg: StubJudgeBackend() if cfg.name == "stub-judge" else StubModelBackend()
                )

                responses_path = run_generation(config)
                judged_path = run_judging(config, responses_path=responses_path)
                payload = score_judged_file(judged_path, output_dir, "smoke")
            finally:
                generate_module.create_backend = original_generate_factory
                judge_module.create_backend = original_judge_factory

            self.assertTrue(responses_path.exists())
            self.assertTrue(judged_path.exists())
            self.assertIn("overall", payload)

            generation_manifest = output_dir / "manifests" / "smoke.generation.json"
            generation_status = output_dir / "manifests" / "smoke.generation.status.json"
            judging_manifest = output_dir / "manifests" / "stub-model__stub-judge.judging.json"
            judging_status = output_dir / "manifests" / "stub-model__stub-judge.judging.status.json"
            self.assertTrue(generation_manifest.exists())
            self.assertTrue(generation_status.exists())
            self.assertTrue(judging_manifest.exists())
            self.assertTrue(judging_status.exists())

            overall_path = output_dir / "stats" / "smoke_overall.json"
            false_compliance_path = output_dir / "stats" / "smoke_false_compliance.json"
            self.assertTrue(overall_path.exists())
            self.assertTrue(false_compliance_path.exists())

            overall = json.loads(overall_path.read_text(encoding="utf-8"))
            self.assertIn("confusion_matrix", overall)

            judged_rows = [
                json.loads(line) for line in judged_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            self.assertTrue(all(row.get("judge_explanation") for row in judged_rows))

    def test_resume_retries_error_rows(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = ModelConfig(name="resume-model", backend="openai", model="stub", api_key="test-key")
            config = RunConfig(
                dataset_path=self.dataset_path,
                output_dir=output_dir,
                limit=2,
                model=model,
                run_name="resume-errors",
            )

            from eval_harness.backends.base import BackendResult
            from eval_harness import generate as generate_module

            class FailThenSucceedBackend:
                def __init__(self) -> None:
                    self.calls = 0

                def generate(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    self.calls += 1
                    if self.calls <= 2:
                        raise RuntimeError("hard failure")
                    return BackendResult(text="ok-after-retry", metadata={})

            backend = FailThenSucceedBackend()
            original_factory = generate_module.create_backend
            try:
                generate_module.create_backend = lambda cfg: backend
                first_path = run_generation(config)
                first_rows = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]
                self.assertTrue(all(row["status"] == "error" for row in first_rows))

                second_path = run_generation(config)
                second_rows = [json.loads(line) for line in second_path.read_text(encoding="utf-8").splitlines()]
            finally:
                generate_module.create_backend = original_factory

            self.assertEqual(second_path, first_path)
            self.assertEqual(len(second_rows), 2)
            self.assertTrue(all(row["status"] == "success" for row in second_rows))
            self.assertTrue(all(row["response"] == "ok-after-retry" for row in second_rows))
            status = json.loads(
                (output_dir / "manifests" / "resume-errors.generation.status.json").read_text(encoding="utf-8")
            )
            self.assertTrue(status["complete"])

    def test_generation_retries_transient_backend_failures(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = ModelConfig(
                name="retry-model",
                backend="openai",
                model="stub",
                api_key="test-key",
                max_retries=1,
                retry_base_delay_seconds=0.0,
                retry_max_delay_seconds=0.0,
                retry_jitter_seconds=0.0,
            )
            config = RunConfig(
                dataset_path=self.dataset_path,
                output_dir=output_dir,
                limit=2,
                model=model,
                run_name="retry-smoke",
            )

            from eval_harness.backends.base import BackendResult
            from eval_harness.backends.base import InferenceBackend
            from eval_harness import generate as generate_module

            class FlakyBackend(InferenceBackend):
                def __init__(self) -> None:
                    super().__init__(
                        model="stub",
                        max_retries=1,
                        retry_base_delay_seconds=0.0,
                        retry_max_delay_seconds=0.0,
                        retry_jitter_seconds=0.0,
                    )
                    self.calls = {}

                def _generate_once(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    count = self.calls.get(prompt, 0)
                    self.calls[prompt] = count + 1
                    if count == 0:
                        raise RuntimeError("transient timeout")
                    return BackendResult(text="ok", metadata={"attempt": count + 1})

            original_factory = generate_module.create_backend
            try:
                generate_module.create_backend = lambda cfg: FlakyBackend()
                responses_path = run_generation(config)
            finally:
                generate_module.create_backend = original_factory

            rows = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(row["status"] == "success" for row in rows))
            self.assertTrue(all(row["response"] == "ok" for row in rows))

    def test_plan_command_helpers_and_csv_exports(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = ModelConfig(name="plan-model", backend="openai", model="stub", api_key="test-key")
            judge = JudgeConfig(
                name="plan-judge",
                backend="openai",
                model="judge-stub",
                api_key="test-key",
                prompt_template_path=self.judge_template_path,
            )
            config = RunConfig(
                dataset_path=self.dataset_path,
                output_dir=output_dir,
                limit=3,
                model=model,
                judge=judge,
                run_name="plan-smoke",
            )

            generation_plan = plan_generation(config)
            self.assertEqual(generation_plan["task"], "generation_plan")
            self.assertEqual(generation_plan["dataset"]["n_rows"], 3)
            self.assertIn("output_path", generation_plan)
            self.assertIn("fingerprints", generation_plan)

            from eval_harness.backends.base import BackendResult
            from eval_harness import generate as generate_module
            from eval_harness import judge as judge_module

            class StubModelBackend:
                def generate(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    return BackendResult(text="response", metadata={})

            class StubJudgeBackend:
                def generate(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    return BackendResult(text="complied", metadata={})

            original_generate_factory = generate_module.create_backend
            original_judge_factory = judge_module.create_backend
            try:
                generate_module.create_backend = lambda cfg: StubModelBackend()
                judge_module.create_backend = lambda cfg: StubJudgeBackend()
                responses_path = run_generation(config)
                judging_plan = plan_judging(config, responses_path=responses_path)
                self.assertTrue(judging_plan["responses_exist"])
                judged_path = run_judging(config, responses_path=responses_path)
                score_payload = score_judged_file(judged_path, output_dir, "plan-smoke")
            finally:
                generate_module.create_backend = original_generate_factory
                judge_module.create_backend = original_judge_factory

            self.assertTrue((output_dir / "stats" / "plan-smoke_per_category.csv").exists())
            self.assertTrue((output_dir / "stats" / "plan-smoke_per_language.csv").exists())
            self.assertTrue((output_dir / "stats" / "plan-smoke_mt_split.csv").exists())
            self.assertTrue((output_dir / "stats" / "plan-smoke_summary_index.json").exists())
            self.assertEqual(score_payload["run_stem"], "plan-smoke")

    def test_load_run_config_from_toml_with_aliases(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "eval_harness" / "examples" / "baseline_gpt4o.toml"
        config = load_run_config(config_path)

        self.assertEqual(config.resolved_run_name, "baseline_gpt4o")
        self.assertIsNotNone(config.model)
        self.assertEqual(config.model.name, "gpt-4o-baseline")
        self.assertEqual(config.model.model, "gpt-4o")
        self.assertIsNotNone(config.judge)
        self.assertEqual(config.judge.name, "gpt-4o-mini-judge")
        self.assertIn("config_path", config.extra_metadata)
        self.assertIn("model_alias", config.extra_metadata)
        self.assertTrue(config_fingerprint(config_path))

    def test_empty_generation_is_recorded_as_failure_and_excluded_from_metrics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = ModelConfig(name="empty-model", backend="openai", model="stub", api_key="test-key")
            judge = JudgeConfig(
                name="judge",
                backend="openai",
                model="judge-stub",
                api_key="test-key",
                prompt_template_path=self.judge_template_path,
            )
            config = RunConfig(
                dataset_path=self.dataset_path,
                output_dir=output_dir,
                limit=1,
                model=model,
                judge=judge,
                run_name="empty-case",
            )

            from eval_harness.backends.base import BackendResult
            from eval_harness import generate as generate_module
            from eval_harness import judge as judge_module

            class EmptyBackend:
                def generate(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    return BackendResult(text="", metadata={})

            class JudgeBackend:
                def generate(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    return BackendResult(text="complied", metadata={})

            original_generate_factory = generate_module.create_backend
            original_judge_factory = judge_module.create_backend
            try:
                generate_module.create_backend = lambda cfg: EmptyBackend()
                judge_module.create_backend = lambda cfg: JudgeBackend()
                responses_path = run_generation(config)
                judged_path = run_judging(config, responses_path=responses_path)
                payload = score_judged_file(judged_path, output_dir, "empty-case")
            finally:
                generate_module.create_backend = original_generate_factory
                judge_module.create_backend = original_judge_factory

            response_row = json.loads(responses_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(response_row["status"], "error")
            self.assertIn("Empty completion", response_row["error"])

            judged_row = json.loads(judged_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(judged_row["parsing_status"], "generation_error")
            self.assertEqual(payload["n_scored"], 0)
            self.assertEqual(payload["n_unscored"], 1)

    def test_validate_run_artifacts_detects_valid_and_invalid_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = ModelConfig(name="stub-model", backend="openai", model="stub", api_key="test-key")
            judge = JudgeConfig(
                name="stub-judge",
                backend="openai",
                model="stub-judge",
                api_key="test-key",
                prompt_template_path=self.judge_template_path,
            )
            config = RunConfig(
                dataset_path=self.dataset_path,
                output_dir=output_dir,
                limit=2,
                model=model,
                judge=judge,
                run_name="validate-smoke",
            )

            from eval_harness.backends.base import BackendResult
            from eval_harness import generate as generate_module
            from eval_harness import judge as judge_module

            class StubModelBackend:
                def generate(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    return BackendResult(text=f"response::{prompt[:10]}", metadata={})

            class StubJudgeBackend:
                def generate(self, prompt: str, system_prompt: str | None = None) -> BackendResult:
                    return BackendResult(text="complied", metadata={})

            original_generate_factory = generate_module.create_backend
            original_judge_factory = judge_module.create_backend
            try:
                generate_module.create_backend = lambda cfg: StubModelBackend()
                judge_module.create_backend = lambda cfg: StubJudgeBackend()
                responses_path = run_generation(config)
                judged_path = run_judging(config, responses_path=responses_path)
            finally:
                generate_module.create_backend = original_generate_factory
                judge_module.create_backend = original_judge_factory

            generation_manifest = output_dir / "manifests" / "validate-smoke.generation.json"
            judging_manifest = output_dir / "manifests" / "stub-model__stub-judge.judging.json"
            validation = validate_run_artifacts(
                responses_path=responses_path,
                judged_path=judged_path,
                generation_manifest_path=generation_manifest,
                judging_manifest_path=judging_manifest,
            )
            self.assertTrue(validation["ok"])
            self.assertEqual(validation["issues"], [])

            bad_lines = responses_path.read_text(encoding="utf-8").splitlines()
            bad_row = json.loads(bad_lines[0])
            bad_row["response"] = ""
            bad_lines[0] = json.dumps(bad_row)
            responses_path.write_text("\n".join(bad_lines) + "\n", encoding="utf-8")

            broken_validation = validate_run_artifacts(responses_path=responses_path)
            self.assertFalse(broken_validation["ok"])
            self.assertTrue(any("empty response" in issue for issue in broken_validation["issues"]))

    def test_config_rejects_invalid_release_facing_argument_combinations(self) -> None:
        with self.assertRaises(ValueError):
            RunConfig(dataset_path=self.dataset_path, limit=10, random_sample=5)

        with self.assertRaises(ValueError):
            RunConfig(
                dataset_path=self.dataset_path,
                language_include=("english",),
                language_exclude=("english",),
            )

        with self.assertRaises(ValueError):
            ModelConfig(name="missing-key", backend="openai", model="gpt-4o", api_key=None)

        with self.assertRaises(ValueError):
            ModelConfig(
                name="missing-url",
                backend="api",
                provider="openrouter",
                model="openai/gpt-4o-mini",
                api_key="key",
                base_url=None,
            )

    def test_new_backend_provider_mode_mapping(self) -> None:
        api_openai = ModelConfig(
            name="api-openai",
            backend="api",
            provider="openai",
            model="gpt-4o-mini",
            api_key="key",
        )
        self.assertEqual(api_openai.resolved_backend_label, "openai")

        api_openrouter = ModelConfig(
            name="api-openrouter",
            backend="api",
            provider="openrouter",
            model="openai/gpt-4o-mini",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )
        self.assertEqual(api_openrouter.resolved_backend_label, "openai_compatible")

        vllm_local = ModelConfig(
            name="vllm-local",
            backend="vllm",
            mode="local",
            model="meta-llama/Llama-3.1-8B-Instruct",
        )
        self.assertEqual(vllm_local.resolved_backend_label, "vllm")

        vllm_server = ModelConfig(
            name="vllm-server",
            backend="vllm",
            mode="server",
            model="meta-llama/Llama-3.1-8B-Instruct",
            api_key="dummy",
            base_url="http://localhost:8000/v1",
        )
        self.assertEqual(vllm_server.resolved_backend_label, "vllm_openai")

        vllm_default = ModelConfig(
            name="vllm-default",
            backend="vllm",
            model="meta-llama/Llama-3.1-8B-Instruct",
            api_key="dummy",
            base_url="http://localhost:8000/v1",
        )
        self.assertEqual(vllm_default.mode, "server")
        self.assertEqual(vllm_default.resolved_backend_label, "vllm_openai")

    def test_validate_run_cli_returns_nonzero_on_invalid_artifacts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            bad_responses = Path(tmp_dir) / "bad.jsonl"
            bad_responses.write_text(
                json.dumps(
                    {
                        "row_id": "r1",
                        "prompt": "hello",
                        "status": "success",
                        "response": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "eval_harness.cli",
                    "validate-run",
                    "--responses-path",
                    str(bad_responses),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn('"ok": false', result.stdout.lower())


if __name__ == "__main__":
    unittest.main()