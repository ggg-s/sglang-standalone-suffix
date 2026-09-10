"""Paired suffix-first A/B measurements, with explicit draft-skip evidence."""
import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from summarize_dynamic_k_experiment import (
    METRICS, SKIP_DRAFT_METRICS, parse_measurement_log, read_snapshot, subtract,
)


def summarize(root):
    groups = defaultdict(dict)
    for path in sorted(root.iterdir()):
        match = re.fullmatch(r"r(\d+)_(baseline|skip)_bs(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        repeat, mode, bs = match.groups()
        run = path / "dynamic_k4_k16"
        logs = list(run.glob(f"measurement_bs{bs}_n*.log"))
        row = parse_measurement_log(logs[0]) if len(logs) == 1 else None
        if row is None or row["successful_requests"] != row["submitted_requests"]:
            raise ValueError(f"Incomplete measurement: {path}")
        required = METRICS + SKIP_DRAFT_METRICS
        delta = subtract(
            read_snapshot(run / f"metrics_after_measurement_bs{bs}.prom", required_metrics=required),
            read_snapshot(run / "metrics_after_k8_probe.prom", required_metrics=required),
        )
        row["skipped"] = delta[SKIP_DRAFT_METRICS[0]]
        if mode == "baseline" and row["skipped"] != 0:
            raise ValueError(f"Baseline unexpectedly skipped draft: {path}")
        if mode == "skip" and row["skipped"] <= 0:
            raise ValueError(f"No draft skips; optimization was not exercised: {path}")
        groups[int(bs)][(int(repeat), mode)] = row
    if not groups:
        raise ValueError("No suffix draft A/B results")
    lines = [
        "| Concurrency | Repeats | Baseline tok/s | Skip tok/s | Delta | Baseline / skip TTFT ms | Baseline / skip TPOT ms | Skipped request steps |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bs, runs in sorted(groups.items()):
        repeats = sorted({key[0] for key in runs})
        if any((r, m) not in runs for r in repeats for m in ("baseline", "skip")):
            raise ValueError(f"Missing A/B pair for concurrency {bs}")
        base = [runs[r, "baseline"] for r in repeats]
        skip = [runs[r, "skip"] for r in repeats]
        med = lambda rows, key: median(row[key] for row in rows)
        b, s = med(base, "total_output_tok_s"), med(skip, "total_output_tok_s")
        if b <= 0:
            raise ValueError("Invalid baseline throughput")
        lines.append(
            f"| {bs} | {len(repeats)} | {b:.2f} | {s:.2f} | {s/b-1:+.2%} | "
            f"{med(base, 'mean_ttft_ms'):.2f} / {med(skip, 'mean_ttft_ms'):.2f} | "
            f"{med(base, 'mean_tpot_ms'):.2f} / {med(skip, 'mean_tpot_ms'):.2f} | "
            f"{sum(row['skipped'] for row in skip):.0f} |"
        )
    lines.append("\nThroughput/latency are medians; skips sum measurement-only request steps, not unique requests.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    print(summarize(parser.parse_args().results_dir))
