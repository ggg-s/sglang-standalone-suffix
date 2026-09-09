#!/usr/bin/env bash
# Run in the existing GPU Python environment with port 30000 available.
# Sync the gq112 fork checkout from ggg-s, check correctness, then benchmark.
# Paths and benchmark settings can be overridden through environment variables.
set -Eeuo pipefail

# Keep origin pointing to the fork; fetch the upstream URL directly.
export SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SGLANG_DIR}"
git switch main
git fetch https://github.com/ggg-s/sglang-standalone-suffix.git main
git merge --ff-only FETCH_HEAD

export PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

source "${SGLANG_DIR}/scripts/configure_cpp_runtime.sh"
configure_cpp_runtime
check_zmq_runtime

export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export TP_SIZE="${TP_SIZE:-4}"
export MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen2.5-72B-Instruct-AWQ}"
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/Qwen/Qwen3-0.6B}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

export SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
export TEST_SCRIPT="${TEST_SCRIPT:-${SPEC_FORGE_DIR}/test_req.py}"
export DATASET_NAME="${DATASET_NAME:-openai-chat}"
export DATASET_PATH="${DATASET_PATH:-${SPEC_FORGE_DIR}/donghuayiwei_fixed.jsonl}"

export ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.72}"
export PORT="${PORT:-30000}"
export CLIENT_BASE_URL="${CLIENT_BASE_URL:-http://127.0.0.1:${PORT}}"

export CONCURRENCIES="${CONCURRENCIES:-20 24 30}"
export REPEATS="${REPEATS:-3}"
export FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN:-2048}"
export RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/ragged_bucket_ab_$(date +%Y%m%d_%H%M%S)}"

for required_file in "${TEST_SCRIPT}" "${DATASET_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file does not exist: ${required_file}" >&2
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python test/srt/test_ragged_cuda_graph.py -v
python test/srt/test_ragged_bucket_summary.py -v

# The A/B runner starts, warms up, measures and stops each server itself.
bash scripts/run_ragged_bucket_ab.sh
cat "${RESULTS_DIR}/ragged_bucket_ab_summary.md"
