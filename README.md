# EU-Guard Eval Harness

Evaluation scaffold for:
- **generation via API backends**
- **generation via vLLM**
- **LLM-as-a-Judge scoring** with both **API judges** and **open-source judges**

## Supported backend interface

### Preferred CLI shape
- `--backend api --provider openai`
- `--backend api --provider anthropic`
- `--backend api --provider together`
- `--backend api --provider openrouter`
- `--backend vllm --mode local`
- `--backend vllm --mode server`

### What the vLLM modes mean
- `--backend vllm` or `--backend vllm --mode server`: call a **separately running vLLM HTTP server** over an OpenAI-compatible API. This is now the default and recommended path.
- `--backend vllm --mode local`: load and run vLLM **inside the eval Python process**.

Legacy backend labels (`openai`, `anthropic`, `together`, `openai_compatible`, `vllm`, `vllm_openai`) are still accepted for compatibility.

The same backend abstraction is used for both the model under test and the judge.

## Installation

This repo can now be used as a Python project via `uv` or standard `pip`.

## Dataset setup

The eval harness expects the benchmark data under:

```text
eval_harness/data/EU_alert_working_copy/test/test.jsonl
```

Create the local data directory and download the gated dataset from Hugging Face:

```bash
mkdir -p data
git lfs install
mkdir -p eval_harness/data
git clone https://huggingface.co/datasets/EU-Guard/EU_alert_working_copy eval_harness/data/EU_alert_working_copy
```

Notes:

- You must have access to the gated dataset before cloning it.
- If needed, log in first with `huggingface-cli login`.
- The default CLI dataset path now points to `eval_harness/data/EU_alert_working_copy/test/test.jsonl`.

### Using uv

Install core dependencies:

```bash
uv sync
```

Install with local vLLM support:

```bash
uv sync --extra vllm
```

Install with developer tooling:

```bash
uv sync --extra dev
```

Run the CLI through uv:

```bash
uv run eval-harness --help
```

### Using pip

Minimum runtime dependencies depend on backend choice:

- `openai` package for `--backend api --provider openai|together|openrouter`
- `anthropic` package for `--backend api --provider anthropic`
- `vllm` package for `--backend vllm --mode local` or for running the helper vLLM server scripts

## vLLM server workflow

Server mode is now the default for `--backend vllm` because it is the most reproducible operational path.

Helper scripts are provided:

- `eval_harness/scripts/start_vllm_model_server.sh`
- `eval_harness/scripts/start_vllm_judge_server.sh`

These scripts intentionally borrow a few research-integrity and scalability defaults from the larger `vllm_script.sh` pattern, without taking on its full Docker/tool-calling complexity:

- explicit `--served-model-name` support for cleaner artifact/accounting consistency
- forced `temperature=0.0` at the server layer for reproducibility
- tunable tensor/data parallel settings for scaling experiments
- tunable GPU memory utilization and attention backend
- consistent timestamped log files under `eval/logs/`
- explicit HF cache path control via environment variables
- optional request logging for audit/debug traces
- prefix caching disabled by default to reduce hidden serving-state variability across research runs

### Start a model-generation server

```bash
bash eval_harness/scripts/start_vllm_model_server.sh
```

Optional overrides:

```bash
VLLM_PORT=8000 \
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
VLLM_TP_SIZE=2 \
VLLM_GPU_MEM_UTIL=0.9 \
bash eval_harness/scripts/start_vllm_model_server.sh
```

Or pass the model name as the first positional argument:

```bash
bash eval_harness/scripts/start_vllm_model_server.sh meta-llama/Llama-3.1-8B-Instruct
```

### Start a judge server

```bash
bash eval_harness/scripts/start_vllm_judge_server.sh
```

Optional overrides:

```bash
VLLM_JUDGE_PORT=8001 \
VLLM_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct \
VLLM_JUDGE_TP_SIZE=2 \
VLLM_JUDGE_GPU_MEM_UTIL=0.9 \
bash eval_harness/scripts/start_vllm_judge_server.sh
```

### Recommended high-value script knobs

For model serving:

- `VLLM_SERVED_MODEL_NAME`: stable API-visible name for cleaner manifests and comparisons
- `VLLM_TP_SIZE` / `VLLM_DP_SIZE`: scale across GPUs when needed
- `VLLM_GPU_MEM_UTIL`: tune memory headroom vs throughput
- `VLLM_ATTN_BACKEND`: choose backend explicitly for repeatability
- `VLLM_LOG_FILE` or `VLLM_LOG_DIR`: preserve server logs for auditability
- `VLLM_CACHE_PATH`: isolate Hugging Face cache location (defaults to `HF_HOME`, then `HOME/.cache`)

For judge serving, the equivalent `VLLM_JUDGE_*` variables are supported.

These scripts do **not** currently replicate the Docker-wrapper, custom chat-template, tool-parser, or reasoning-parser features from the reference script, because they are not necessary for the eval harness’s current research workflow.

Or pass the judge model name as the first positional argument:

```bash
bash eval_harness/scripts/start_vllm_judge_server.sh Qwen/Qwen2.5-7B-Instruct
```

### Stop the servers

- If running in the foreground: press `Ctrl+C`
- If running under another process manager: stop it there
- If launched in the background manually: use `ps` / `pkill -f vllm.entrypoints.openai.api_server` carefully

### Use the running servers in evals

For model generation:

```bash
python3 -m eval_harness.cli generate \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key dummy
```

For judge generation:

```bash
python3 -m eval_harness.cli judge \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --judge-backend vllm \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --judge-base-url http://localhost:8001/v1 \
  --judge-api-key dummy
```

Example:

```bash
python3 -m pip install -r requirements.txt
```

## R1-ready behavior in this scaffold

- Dataset slices are fingerprinted and recorded in manifests.
- Generation and judging are resumable by `row_id`.
- Transient backend failures are retried with bounded exponential backoff and jitter.
- Empty model completions are treated as failures, not silent successes.
- Scoring excludes unjudged / parse-failed rows from headline metrics and exports parse-failure artifacts separately.
- A `plan` command previews the resolved dataset slice, pending counts, and output paths before running.
- `plan` and run manifests now also expose a config fingerprint when using config-driven runs.
- Stats are exported as both JSON and lightweight CSV tables for easier downstream analysis.
- A summary index JSON is written per scored run to make artifact discovery easier.

## Config-driven runs and alias registry

You can now drive runs from TOML or JSON configs.

- Example run spec: `eval_harness/examples/baseline_gpt4o.toml`
- Example alias registry: `eval_harness/examples/aliases.toml`

Supported ideas:

- `model_alias` and `judge_alias`
- inline `model` and `judge` config blocks
- `alias_path` inside the run spec or `--alias-file` on CLI
- CLI overrides for dataset path, output dir, sampling, concurrency, overwrite, and verified-only mode

Example:

```bash
python3 -m eval_harness.cli plan \
  --config eval_harness/examples/baseline_gpt4o.toml
```

```bash
python3 -m eval_harness.cli run-all \
  --config eval_harness/examples/baseline_gpt4o.toml
```

## Output layout

```text
eval/
├── manifests/
├── responses/
├── judged/
└── stats/
```

## CLI

### 1. Generate responses via API

```bash
python3 -m eval_harness.cli generate \
  --dataset-path eval_harness/data/EU_alert_working_copy/test/test.jsonl \
  --output-dir eval \
  --model-name gpt-4o \
  --backend api \
  --provider openai \
  --model gpt-4o \
  --api-key "$OPENAI_API_KEY" \
  --concurrency 8
```

### 2. Generate responses via vLLM server

```bash
python3 -m eval_harness.cli generate \
  --dataset-path eval_harness/data/EU_alert_working_copy/test/test.jsonl \
  --output-dir eval \
  --model-name llama-3.1-8b-instruct \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key dummy
```

`--mode server` is optional because server mode is the default for `--backend vllm`.

### 3. Judge with API judge

```bash
python3 -m eval_harness.cli judge \
  --dataset-path eval_harness/data/EU_alert_working_copy/test/test.jsonl \
  --output-dir eval \
  --model-name gpt-4o \
  --backend api \
  --provider openai \
  --model gpt-4o \
  --judge-name gpt-4o-mini-judge \
  --judge-backend api \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --judge-api-key "$OPENAI_API_KEY" \
  --judge-template-path eval_harness/prompts/judge_refusal.txt
```

### 4. Judge with open-source judge over vLLM

```bash
python3 -m eval_harness.cli judge \
  --dataset-path eval_harness/data/EU_alert_working_copy/test/test.jsonl \
  --output-dir eval \
  --model-name llama-3.1-8b-instruct \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --judge-name qwen-judge \
  --judge-backend vllm \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --judge-base-url http://localhost:8001/v1 \
  --judge-api-key dummy \
  --judge-template-path eval_harness/prompts/judge_refusal.txt
```

Both `--mode server` and `--judge-mode server` are optional because vLLM server mode is now the default.

### 5. End-to-end run

```bash
python3 -m eval_harness.cli run-all \
  --dataset-path eval_harness/data/EU_alert_working_copy/test/test.jsonl \
  --output-dir eval \
  --model-name gpt-4o \
  --backend api \
  --provider openai \
  --model gpt-4o \
  --api-key "$OPENAI_API_KEY" \
  --judge-name gpt-4o-mini-judge \
  --judge-backend api \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --judge-api-key "$OPENAI_API_KEY" \
  --judge-template-path eval_harness/prompts/judge_refusal.txt
```

You can use **OpenRouter** via:

- `--backend api`
- `--provider openrouter`
- `--base-url https://openrouter.ai/api/v1`
- `--api-key "$OPENROUTER_API_KEY"`

### 5.5 Preview a run before execution

```bash
python3 -m eval_harness.cli plan \
  --dataset-path eval_harness/data/EU_alert_working_copy/test/test.jsonl \
  --output-dir eval \
  --model-name gpt-4o \
  --backend api \
  --provider openai \
  --model gpt-4o \
  --api-key "$OPENAI_API_KEY" \
  --judge-name gpt-4o-mini-judge \
  --judge-backend api \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --judge-api-key "$OPENAI_API_KEY" \
  --judge-template-path eval_harness/prompts/judge_refusal.txt
```

### 5.6 End-to-end run from config

```bash
python3 -m eval_harness.cli run-all \
  --config eval_harness/examples/baseline_gpt4o.toml
```

### 6. Score only

```bash
python3 -m eval_harness.cli score \
  --judged-path eval/judged/gpt-4o__gpt-4o-mini-judge.jsonl \
  --output-dir eval \
  --run-stem gpt-4o__gpt-4o-mini-judge
```

### 7. Inspect a dataset slice before running

```bash
python3 -m eval_harness.cli dataset-summary \
  --dataset-path eval_harness/data/EU_alert_working_copy/test/test.jsonl \
  --verified-only \
  --language-include english \
  --limit 20
```

### 8. Validate artifacts after a run

```bash
python3 -m eval_harness.cli validate-run \
  --responses-path eval/responses/gpt-4o.jsonl \
  --judged-path eval/judged/gpt-4o__gpt-4o-mini-judge.jsonl \
  --generation-manifest-path eval/manifests/gpt-4o.generation.json \
  --judging-manifest-path eval/manifests/gpt-4o__gpt-4o-mini-judge.judging.json
```

## Notes

- Responses and judge outputs are **resumable** by `row_id`.
- By default, the harness looks for the dataset at `eval_harness/data/EU_alert_working_copy/test/test.jsonl`.
- `--verified-only` filters to `is_machine_translation=false` rows.
- Dataset slicing also supports `--language-include`, `--language-exclude`, `--category-include`, `--category-exclude`, `--random-sample`, and `--sampling-seed`.
- `--limit` and `--random-sample` are mutually exclusive.
- Include/exclude filters for the same field must not overlap.
- Judge prompting now uses a single prompt template source of truth: `eval_harness/prompts/judge_refusal.txt`.
- `--judge-template-path` is optional; if omitted, the harness defaults to `eval_harness/prompts/judge_refusal.txt`.
- For `--backend vllm` and `--judge-backend vllm`, omitted mode now defaults to `server`.
- Local dependencies such as `openai`, `anthropic`, or `vllm` are loaded lazily at runtime.
- Generation and judging write run manifests and status files under `eval/manifests/`.
- Per-backend reliability controls include `--timeout-seconds`, `--max-retries`, `--retry-base-delay-seconds`, `--retry-max-delay-seconds`, and `--retry-jitter-seconds` plus judge-specific overrides.
- Judge sampling can be overridden independently with `--judge-temperature`, `--judge-top-p`, and `--judge-max-tokens`.
- `validate-run` performs lightweight integrity checks on responses, judged outputs, and manifests before you trust a run for comparison.
- `validate-run` exits non-zero on failure, so it can be used in CI or release checks.
- `plan` is a non-executing preview step intended as an R1-safe workflow check before spending API or local inference time.
- Config-driven runs support TOML or JSON run specs and optional alias registries.
- When scoring completes, a `*_summary_index.json` file is written to summarize the main metric artifacts for the run.
- Scoring writes:
  - overall metrics (JSON)
  - per-category metrics (JSON + CSV)
  - per-language metrics (JSON + CSV)
  - per-annex-section metrics (JSON + CSV)
  - verified-only vs full split (JSON + CSV)
  - false-compliance and over-refusal example exports (JSON + CSV)
  - parse-failure exports when present (JSON + CSV)
  - summary index (JSON)

## Current limitations

- This is still an early harness: it now includes basic retry/backoff/timeout handling, but not provider-specific cost accounting or batch inference APIs.
- Local `vllm` mode now reuses a single in-process engine per backend instance, but server mode remains the recommended default for reproducibility and operations.