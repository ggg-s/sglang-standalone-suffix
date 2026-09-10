#!/usr/bin/env bash
# Identical suffix policy and graphs; change only whether suffix rows skip draft.
set -Eeuo pipefail
export SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
export DATASET_PATH="${DATASET_PATH:-${SPEC_FORGE_DIR}/donghuayiwei_fixed.jsonl}"
export GPU_IDS="${GPU_IDS:-4,5,6,7}"
export TP_SIZE="${TP_SIZE:-4}"
export PORT="${PORT:-30001}"
export CLIENT_BASE_URL="${CLIENT_BASE_URL:-http://127.0.0.1:${PORT}}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.60}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-20}"
export RAGGED_CUDA_GRAPH="${RAGGED_CUDA_GRAPH:-0}"
export RAGGED_GRAPH_MAX_BS="${RAGGED_GRAPH_MAX_BS:-20}"
export SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-3600}"
export EXPERIMENTS=dynamic_k4_k16 DYNAMIC_EXPERIMENT_NAME=dynamic_k4_k16
export DYNAMIC_LONG_DRAFT_TOKENS=16 DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN=23
export HIGH_BS_THRESHOLD=24 SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK=8:23
export SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO=1.0
unset SGLANG_DYNAMIC_K_TIERS SGLANG_DYNAMIC_K_BATCH_POLICY SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS
CONCURRENCIES="${CONCURRENCIES:-20}"
REPEATS="${REPEATS:-3}"
OUTPUT_ROOT="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/suffix_skip_draft_ab_$(date +%Y%m%d_%H%M%S)}"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || { echo 'REPEATS must be positive' >&2; exit 2; }
for concurrency in ${CONCURRENCIES}; do
    [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid concurrency' >&2; exit 2; }
    (( concurrency <= MAX_RUNNING_REQUESTS )) || { echo 'Raise MAX_RUNNING_REQUESTS to cover concurrency' >&2; exit 2; }
done
source "${SGLANG_DIR}/scripts/configure_cpp_runtime.sh"
configure_cpp_runtime
python -c 'from arctic_inference.suffix_decoding import SuffixDecodingCache, SuffixDecodingDraft'
python "${SGLANG_DIR}/test/srt/test_suffix_skip_draft.py"
mkdir -p "${OUTPUT_ROOT}"
for concurrency in ${CONCURRENCIES}; do
    for ((round=1; round<=REPEATS; ++round)); do
        modes=(baseline skip)
        if (( round % 2 == 0 )); then modes=(skip baseline); fi
        for mode in "${modes[@]}"; do
            enabled=0
            if [[ "${mode}" == skip ]]; then enabled=1; fi
            SUFFIX_SKIP_DRAFT="${enabled}" \
            RESULTS_DIR="${OUTPUT_ROOT}/r${round}_${mode}_bs${concurrency}" \
            MEASURE_CONCURRENCY="${concurrency}" \
            MEASURE_PROMPTS="${PROMPTS_PER_CONCURRENCY:-$((concurrency * 4))}" \
            bash "${SGLANG_DIR}/scripts/run_dynamic_k_experiment.sh"
        done
    done
done
python "${SGLANG_DIR}/scripts/summarize_suffix_skip_draft_ab.py" "${OUTPUT_ROOT}" \
    | tee "${OUTPUT_ROOT}/suffix_skip_draft_summary.md"
