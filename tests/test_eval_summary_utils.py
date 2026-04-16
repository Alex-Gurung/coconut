import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_summary_utils import collect_result, discover_eval_results, summary_basename


class EvalSummaryUtilsTest(unittest.TestCase):
    def test_collect_result_uses_saved_config_when_schedule_is_not_provided(self):
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "demo-run-eval-ckpt_004"
            eval_dir.mkdir()
            (eval_dir / "eval_outputs.json").write_text(
                json.dumps(
                    {
                        "accuracy": 0.75,
                        "total_samples": 8,
                        "config": {
                            "epochs_per_stage": 2,
                            "max_latent_stage": 10,
                            "c_thought": 3,
                        },
                        "outputs": [
                            {"gen_tokens": 11},
                            {"gen_tokens": 13},
                        ],
                    }
                )
            )

            row = collect_result(eval_dir, "checkpoint_4", 4)

        self.assertEqual(row["stage"], 2)
        self.assertEqual(row["num_latents"], 6)
        self.assertAlmostEqual(row["accuracy"], 0.75)
        self.assertAlmostEqual(row["avg_gen_tokens"], 12.0)
        self.assertEqual(row["samples"], 8)

    def test_discover_eval_results_filters_suffixes_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_name = "demo-run"

            base_dir = root / f"{run_name}-eval-ckpt_002"
            base_dir.mkdir()
            (base_dir / "eval_outputs.json").write_text(
                json.dumps({"accuracy": 0.2, "total_samples": 5, "outputs": []})
            )

            test_dir = root / f"{run_name}-eval-ckpt_004-test"
            test_dir.mkdir()
            (test_dir / "eval_outputs.json").write_text(
                json.dumps({"accuracy": 0.4, "total_samples": 5, "outputs": []})
            )

            base_rows = discover_eval_results(root, run_name)
            test_rows = discover_eval_results(root, run_name, eval_suffix="test")

        self.assertEqual([row["checkpoint"] for row in base_rows], ["checkpoint_2"])
        self.assertEqual([row["checkpoint"] for row in test_rows], ["checkpoint_4"])

    def test_summary_basename_includes_suffix(self):
        self.assertEqual(summary_basename(None), "eval_summary")
        self.assertEqual(summary_basename("test"), "eval_summary-test")


if __name__ == "__main__":
    unittest.main()
