import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from summarize_dynamic_k_experiment import SKIP_DRAFT_METRICS
from summarize_suffix_skip_draft_ab import summarize
import test_ragged_bucket_summary as fixtures


class SkipSummaryTests(unittest.TestCase):
    def make_run(self, root, mode, skips):
        run = fixtures.BucketSummaryTests().make_run(root, mode, 100 if mode == "baseline" else 110)
        for phase, count in (("after_k8_probe", 20), ("after_measurement_bs10", 20 + skips)):
            with (run / f"metrics_{phase}.prom").open("a") as f:
                for name in SKIP_DRAFT_METRICS:
                    f.write(f'{name}{{tp_rank="0"}} {count}\n')
                    f.write(f'{name}{{tp_rank="1"}} 9999\n')
        return run

    def test_measured_delta_and_pair(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_run(root, "baseline", 0)
            with self.assertRaisesRegex(ValueError, "Missing A/B pair"):
                summarize(root)
            self.make_run(root, "skip", 50)
            report = summarize(root)
            self.assertIn("+10.00%", report)
            self.assertIn("| 50 |", report)

    def test_zero_skips_are_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_run(root, "baseline", 0)
            self.make_run(root, "skip", 0)
            with self.assertRaisesRegex(ValueError, "No draft skips"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
