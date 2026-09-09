#!/usr/bin/env python3
"""Report isolated, paired bucket/eager measurements; fail on incomplete runs."""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from summarize_dynamic_k_experiment import (
    parse_measurement_log,
    read_snapshot,
    subtract,
)


def summarize(root):
    groups = defaultdict(dict)
    for path in sorted(root.iterdir()):
        match = re.fullmatch(r"r(\d+)_(eager|bucket)_bs(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        repeat, mode, bs = match.groups()
        run = path / "dynamic_k4_k16"
        logs = list(run.glob(f"measurement_bs{bs}_n*.log"))
        row = parse_measurement_log(logs[0]) if len(logs) == 1 else None
        if row is None or row["successful_requests"] != row["submitted_requests"]:
            raise ValueError(f"Incomplete or failed measurement: {path}")
        before = run / "metrics_after_k8_probe.prom"
        after = run / f"metrics_after_measurement_bs{bs}.prom"
        if not before.exists() or not after.exists():
            raise ValueError(f"Missing per-phase metrics: {path}")
        delta = subtract(read_snapshot(after), read_snapshot(before))
        row["graph_batches"] = delta[
            "sglang:ragged_verify_bucket_cuda_graph_batch_total"
        ]
        row["real_tokens"] = delta["sglang:ragged_verify_bucket_real_token_total"]
        row["padding_tokens"] = delta["sglang:ragged_verify_bucket_padding_token_total"]
        groups[int(bs)][(int(repeat), mode)] = row
    if not groups:
        raise ValueError("No bucket/eager measurements found")
    lines = [
        "| Concurrency | Repeats | Eager tok/s | Bucket tok/s | Delta | Eager / bucket TTFT ms | Eager / bucket TPOT ms | Bucket replays | Padding / real tokens |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bs, runs in sorted(groups.items()):
        repeats = sorted({key[0] for key in runs})
        for repeat in repeats:
            if any((repeat, mode) not in runs for mode in ("eager", "bucket")):
                raise ValueError(f"Missing A/B pair: B={bs}, repeat={repeat}")
        eager = [runs[(repeat, "eager")] for repeat in repeats]
        bucket = [runs[(repeat, "bucket")] for repeat in repeats]
        med = lambda rows, key: median(row[key] for row in rows)
        base = med(eager, "total_output_tok_s")
        candidate = med(bucket, "total_output_tok_s")
        if base <= 0:
            raise ValueError(f"Invalid baseline throughput: B={bs}")
        real = sum(row["real_tokens"] for row in bucket)
        padding = sum(row["padding_tokens"] for row in bucket)
        ratio = f"{padding / real:.2%}" if real else "n/a"
        lines.append(
            f"| {bs} | {len(repeats)} | {base:.2f} | {candidate:.2f} | "
            f"{candidate / base - 1:+.2%} | "
            f"{med(eager, 'mean_ttft_ms'):.2f} / {med(bucket, 'mean_ttft_ms'):.2f} | "
            f"{med(eager, 'mean_tpot_ms'):.2f} / {med(bucket, 'mean_tpot_ms'):.2f} | "
            f"{sum(row['graph_batches'] for row in bucket):.0f} | {ratio} |"
        )
    lines.append(
        "\nThroughput and latency are medians; replay/token counters sum measurement-only deltas. Zero replays do not validate bucket acceleration."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    print(summarize(args.results_dir))
