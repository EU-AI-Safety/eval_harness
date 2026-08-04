#!/usr/bin/env bash
# Run on a JUWELS login node after joining laionize (or transfernetx).
# Creates a shared mamba env under project scratch to avoid home inode limits.
#
# Usage:
#   bash eval_harness/scripts/hpc/setup_env_juwels.sh
#
# Override defaults:
#   PROJECT_ACCOUNT=laionize EU_GUARD_ROOT=/p/project1/laionize/$USER/eu-guard \
#     bash eval_harness/scripts/hpc/setup_env_juwels.sh

set -euo pipefail

PROJECT_ACCOUNT="${PROJECT_ACCOUNT:-laionize}"
MACHINE="${MACHINE:-juwels}"
NAME_ENV="${NAME_ENV:-eu_guard_eval}"
EU_GUARD_ROOT="${EU_GUARD_ROOT:-/p/project1/${PROJECT_ACCOUNT}/${USER}/eu-guard}"

USER_SHARED="/p/scratch/${PROJECT_ACCOUNT}/${USER}_${MACHINE}_shared"
SHARED_MAMBA_ENV="${USER_SHARED}/mamba/base_${NAME_ENV}"
SHARED_ENV="${USER_SHARED}/envs/mamba_${NAME_ENV}"
HF_CACHE="${USER_SHARED}/hf_cache"

echo "Project account : ${PROJECT_ACCOUNT}"
echo "EU-Guard root   : ${EU_GUARD_ROOT}"
echo "Mamba base      : ${SHARED_MAMBA_ENV}"
echo "Mamba env       : ${SHARED_ENV}"
echo "HF cache        : ${HF_CACHE}"

module --force purge

mkdir -p "${USER_SHARED}/tmp" "${HF_CACHE}" "$(dirname "${EU_GUARD_ROOT}")"

if [[ ! -d "${SHARED_MAMBA_ENV}" ]]; then
  cd "${USER_SHARED}/tmp"
  curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  chmod +x "Miniforge3-$(uname)-$(uname -m).sh"
  bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "${SHARED_MAMBA_ENV}"
fi

eval "$("${SHARED_MAMBA_ENV}/bin/conda" shell.bash hook)"

if [[ ! -d "${SHARED_ENV}" ]]; then
  mamba create --prefix "${SHARED_ENV}" --channel conda-forge --override-channels -y
fi

source "${SHARED_MAMBA_ENV}/bin/activate" "${SHARED_ENV}"

mamba install -y python=3.12 pip -c conda-forge --override-channels

if [[ ! -d "${EU_GUARD_ROOT}/eval_harness" ]]; then
  echo "Clone EU-Guard into ${EU_GUARD_ROOT} (login node has internet):"
  echo "  git clone <your-repo-url> ${EU_GUARD_ROOT}"
  exit 1
fi

PACKAGE_ROOT="${EU_GUARD_ROOT}"
if [[ ! -f "${PACKAGE_ROOT}/pyproject.toml" && -f "${EU_GUARD_ROOT}/eval_harness/pyproject.toml" ]]; then
  PACKAGE_ROOT="${EU_GUARD_ROOT}/eval_harness"
fi

if [[ ! -f "${PACKAGE_ROOT}/pyproject.toml" ]]; then
  echo "No pyproject.toml found in ${EU_GUARD_ROOT} or ${EU_GUARD_ROOT}/eval_harness." >&2
  echo "Pull the latest eval_harness checkout before rerunning this script." >&2
  exit 1
fi

pip install -e "${PACKAGE_ROOT}[vllm]"

cat <<EOF

Environment ready. Activate with:

  module --force purge
  source ${SHARED_MAMBA_ENV}/bin/activate ${SHARED_ENV}
  export HF_HOME=${HF_CACHE}
  cd ${EU_GUARD_ROOT}

Download gated benchmark data on the login node:

  huggingface-cli login
  git lfs install
  git clone https://huggingface.co/datasets/EU-Guard/EU_alert_working_copy \\
    ${EU_GUARD_ROOT}/eval_harness/data/EU_alert_working_copy

Prefetch models on the login node into THIS same cache path
(compute nodes have no internet; sbatch jobs set HF_HUB_OFFLINE=1):

  export HF_HOME=${HF_CACHE}
  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
  huggingface-cli download Qwen/Qwen2.5-7B-Instruct

Before first smoke submit:

  cd ${EU_GUARD_ROOT}
  mkdir -p eval/logs
  sbatch eval_harness/scripts/hpc/sbatch_smoke_juwels.sh

EOF
