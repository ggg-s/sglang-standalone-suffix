import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from summarize_dynamic_k_experiment import METRICS, read_snapshot
from summarize_ragged_bucket_ab import summarize


class BucketSummaryTests(unittest.TestCase):
    def make_run(self, root, mode, throughput, completed=40):
        run = root / f"r1_{mode}_bs10" / "dynamic_k4_k16"
        run.mkdir(parents=True)
        (run / "measurement_bs10_n40.log").write_text(
            f"Successful requests: {completed}/40\n"
            "Benchmark duration (s): 20\n"
            "Mean decoding throughput (tok/s): 10\n"
            "Mean output throughput (tok/s): 10\n"
            f"Total E2E output throughput (tok/s): {throughput}\n"
            "Mean TTFT (ms): 100\nMean TPOT (ms): 20\nMean ITL (ms): 30\n"
        )

        def metrics(batch, real, padding):
            values = dict.fromkeys(METRICS, 0)
            values.update(
                {
                    "sglang:ragged_verify_bucket_cuda_graph_batch_total": batch,
                    "sglang:ragged_verify_bucket_real_token_total": real,
                    "sglang:ragged_verify_bucket_padding_token_total": padding,
                }
            )
            return "".join(
                f'{name}{{tp_rank="0"}} {value}\n'
                f'{name}{{tp_rank="1"}} {value * 100}\n'
                for name, value in values.items()
            )

        (run / "metrics_after_k8_probe.prom").write_text(metrics(100, 10000, 1000))
        (run / "metrics_after_measurement_bs10.prom").write_text(
            metrics(102, 10160, 1008)
        )
        return run

    def test_measurement_only_tp0_deltas_and_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root, "eager", 100)
            self.make_run(root, "bucket", 105)
            result = summarize(root)
            self.assertIn("| 10 | 1 | 100.00 | 105.00 | +5.00% |", result)
            self.assertIn("| 2 | 5.00% |", result)

    def test_missing_pair_or_failed_requests_are_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root, "eager", 100)
            with self.assertRaisesRegex(ValueError, "Missing A/B pair"):
                summarize(root)
            self.make_run(root, "bucket", 110, completed=39)
            with self.assertRaisesRegex(ValueError, "Incomplete or failed"):
                summarize(root)

    def test_missing_custom_metrics_are_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root, "eager", 100)
            run = self.make_run(root, "bucket", 105)
            snapshot = run / "metrics_after_measurement_bs10.prom"
            # Reproduce a valid scrape with tokenizer counters but no custom samples.
            snapshot.write_text(
                'sglang:prompt_tokens_total{model_name="model"} 1552380\n'
            )
            with self.assertRaisesRegex(ValueError, "Absent metrics are not zero"):
                summarize(root)

    def test_explicit_zero_is_valid_but_wrong_rank_or_comments_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "metrics.prom"
            metric = METRICS[0]
            for labels in ('{tp_rank="0"}', ""):
                snapshot.write_text(f"{metric}{labels} 0\n")
                self.assertEqual(
                    read_snapshot(snapshot, required_metrics=[metric])[metric], 0
                )
            for content in (
                f'{metric}{{tp_rank="1"}} 3\n',
                f"# HELP {metric} help\n",
                "",
            ):
                snapshot.write_text(content)
                with self.assertRaisesRegex(ValueError, "Missing TP0/unlabeled"):
                    read_snapshot(snapshot, required_metrics=[metric])


if __name__ == "__main__":
    unittest.main()
