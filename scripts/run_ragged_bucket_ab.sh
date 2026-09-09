#!/usr/bin/env bash
# Isolate bucket-graph benefit: identical match>=23 policy, fresh server per
# concurrency and repeat, alternating eager/bucket launch order.
set -Eeuo pipefail

SGLANG_DIR="${SGLANG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEC_FORGE_DIR="${SPEC_FORGE_DIR:-/workspace/SpecForge}"
RESULTS_DIR="${RESULTS_DIR:-${SPEC_FORGE_DIR}/results/ragged_bucket_ab_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
CONCURRENCIES="${CONCURRENCIES:-20 24 30}"
if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEATS must be positive" >&2
    exit 2
fi

mkdir -p "${RESULTS_DIR}"
for concurrency in ${CONCURRENCIES}; do
    if ! [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]]; then
        echo "CONCURRENCIES must contain positive integers" >&2
        exit 2
    fi
    for ((round = 1; round <= REPEATS; ++round)); do
        modes=(eager bucket)
        if (( round % 2 == 0 )); then modes=(bucket eager); fi
        for mode in "${modes[@]}"; do
            enabled=0
            if [[ "${mode}" == "bucket" ]]; then enabled=1; fi
            label="r${round}_${mode}_bs${concurrency}"
            env -u SGLANG_RAGGED_VARLEN_CUDA_GRAPH_PATTERNS \
                -u SGLANG_DYNAMIC_K_TIERS -u SGLANG_DYNAMIC_K_BATCH_POLICY \
                SGLANG_DIR="${SGLANG_DIR}" SPEC_FORGE_DIR="${SPEC_FORGE_DIR}" \
                RESULTS_DIR="${RESULTS_DIR}/${label}" \
                GPU_IDS="${GPU_IDS:-0,1,2,3}" TP_SIZE="${TP_SIZE:-4}" \
                MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}" \
                MEASURE_CONCURRENCY="${concurrency}" \
                MEASURE_PROMPTS="${PROMPTS_PER_CONCURRENCY:-$((concurrency * 4))}" \
                EXPERIMENTS=dynamic_k4_k16 DYNAMIC_EXPERIMENT_NAME=dynamic_k4_k16 \
                DYNAMIC_LONG_DRAFT_TOKENS=16 DYNAMIC_LONG_SUFFIX_MIN_MATCH_LEN=23 \
                HIGH_BS_THRESHOLD=24 SGLANG_DYNAMIC_K_HIGH_BATCH_FALLBACK=8:23 \
                SGLANG_RAGGED_CUDA_GRAPH_MIN_LONG_RATIO=1.0 \
                RAGGED_CUDA_GRAPH="${enabled}" \
                bash "${SGLANG_DIR}/scripts/run_dynamic_k_experiment.sh"
        done
    done
done
python "${SGLANG_DIR}/scripts/summarize_ragged_bucket_ab.py" "${RESULTS_DIR}" \
    | tee "${RESULTS_DIR}/ragged_bucket_ab_summary.md"
