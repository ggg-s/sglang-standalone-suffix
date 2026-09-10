# Suffix-first draft generation (experimental)

`--speculative-suffix-skip-draft` queries suffix candidates before draft
generation. Only requests already selected by the existing dynamic-K quality
policy skip generation. The match floor, score threshold, candidate length,
K selection, target verification, and post-verification draft extend are unchanged.

For mixed batches, the draft model runs on a compact batch of fallback requests.
Its token rows are scattered back to their original request order. Full-hit
batches do not call draft preprocessing or draft generation at all. Accepted
tokens still pass through draft extend so that draft KV is ready for later fallback.
This does not skip target verification or all draft-model computation.

The switch defaults off. Supported scope: STANDALONE, greedy FA3, page size 1,
top-k 1, no grammar/logprobs/penalties, no DP attention, DP/PP size 1, no overlap
future state, and decoder-only models. Other paths retain normal drafting.

## Validation

CPU control-flow and mapping tests:

```bash
python test/srt/test_suffix_skip_draft.py
python test/srt/test_suffix_skip_draft_summary.py
```

These tests use model/tensor doubles and do not establish CUDA numerical or KV
correctness. Before deployment, run identical fixed prompts at temperature zero
with the switch off/on, compare complete output token IDs, exercise long suffix
acceptance/rejection followed by fallback, mixed batches, and request completion.
GPU replay/eager execution and the performance gain require server validation.
Small floating-point differences from changed batch shapes are possible; investigate
output differences rather than treating a throughput-only run as correctness proof.

Paired server benchmark (in the GPU environment, with GPUs 4–7 free):

```bash
GPU_IDS=4,5,6,7 MEM_FRACTION_STATIC=0.60 \
MAX_RUNNING_REQUESTS=20 CONCURRENCIES=20 REPEATS=3 \
bash scripts/run_suffix_skip_draft_ab.sh
```

The runner uses the existing model/dataset path overrides. It alternates baseline
and skip launch order, with a fresh server per run. Both use match >=23, K=4/16,
and the same graph settings. Bucket graphs default off to isolate draft skipping;
set `RAGGED_CUDA_GRAPH=1 RAGGED_GRAPH_MAX_BS=20` to compare with bucket graphs on
in both arms. Results are under `suffix_skip_draft_ab_<timestamp>` with
`suffix_skip_draft_summary.md` reporting median throughput/latency and measured
skip counts. The summary rejects incomplete pairs and skip runs with no skips.

For a single existing experiment use `SUFFIX_SKIP_DRAFT=1` with
`run_dynamic_k_experiment.sh`. The new `draft_skipped_requests` and
`draft_skipped_batches` columns report request decode steps and batches, not
unique requests. Metrics are `sglang:suffix_draft_skipped_request_total` and
`sglang:suffix_draft_skipped_batch_total`.

Arctic's native extension is imported before a suffix benchmark starts, exposing
glibc or dependency errors before expensive model loading.
