"""Synthetic, temporary inputs for analysis checks. No NPU measurements."""

import json
import tempfile
import unittest
from pathlib import Path

from analyze import DataError, analyze


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        fields = ("device_model", "driver", "cann", "torch", "torch_npu", "weights_revision",
                  "tutorial_commit", "decoding_policy", "warmup_policy")
        write(self.root / "environment.json", dict.fromkeys(fields, "SYNTHETIC TEST"))
        for load, tokens in (("short", 8), ("long", 32)):
            for repeat in range(1, 4):
                folder = self.root / load / str(repeat)
                folder.mkdir(parents=True)
                rates = []
                for name, fused in (("baseline", False), ("fused", True)):
                    seconds = (tokens + repeat) * (0.75 if fused else 1)
                    rates.append(4 / seconds)
                    write(folder / (name + ".json"), {
                        "model_path": "fixture", "device": "npu:0", "dtype": "bfloat16",
                        "recipe_exe_mode": "eager", "num_prompts": 1, "batch_size": 1,
                        "resolved_attn_implementation": "fixture", "seed": 1, "enable_thinking": False,
                        "input_tokens": tokens, "generated_tokens": 4, "max_new_tokens": 4,
                        "max_input_tokens": 64, "profiler_enabled": False,
                        "rmsnorm_status": {"rmsnorm_fused": fused, "add_rmsnorm_fused": fused},
                        "generation_text_sanity": {"status": "ok"}, "prefill_time_s": seconds / 2,
                        "decode_time_s": seconds / 2, "total_time_s": seconds, "tokens_per_second": rates[-1]})
                    write(folder / (name + ".jsonl"), {"prompt": "TEST " + load,
                          "generated_text": "SYNTHETIC", "generated_tokens": 4})
                write(folder / "comparison.json", {"status": "ok", "baseline_tokens_per_second": rates[0],
                      "optimized_tokens_per_second": rates[1], "precision_check_ok": True,
                      "precision_check_has_warning": False, "precision_check_hard_failed": False,
                      "baseline_optimized_outputs_consistent": True,
                      "baseline_optimized_output_similarity_ratio": 1})

    def change(self, path, key, value):
        file = self.root / "short/1" / path
        data = json.loads(file.read_text(encoding="utf-8"))
        data[key] = value
        write(file, data)

    def test_independent_workload_medians(self):
        rows = analyze(self.root)["workloads"]
        self.assertEqual(rows["short"]["baseline"]["median_s"], 10)
        self.assertEqual(rows["long"]["baseline"]["median_s"], 34)
        self.assertEqual(rows["short"]["median_time_change_pct"], -25)
        self.assertEqual(len(rows["long"]["samples"]), 3)

    def test_warned_quality_retains_samples_but_suppresses_change(self):
        self.change("comparison.json", "precision_check_has_warning", True)
        result = analyze(self.root)["workloads"]["short"]
        self.assertIsNone(result["median_time_change_pct"])
        self.assertFalse(result["samples"][0]["quality"]["pass"])

    def test_different_prompt_is_rejected(self):
        self.change("fused.jsonl", "prompt", "OTHER")
        with self.assertRaisesRegex(DataError, "same nonempty prompt"):
            analyze(self.root)

    def test_wall_time_fallback_is_rejected(self):
        self.change("baseline.json", "decode_time_s", None)
        with self.assertRaisesRegex(DataError, "stage timing"):
            analyze(self.root)

    def test_profiled_result_is_rejected(self):
        self.change("baseline.json", "profiler_enabled", True)
        with self.assertRaisesRegex(DataError, "Profiler"):
            analyze(self.root)

    def test_missing_pair_file_is_rejected(self):
        (self.root / "short/1/fused.json").unlink()
        with self.assertRaisesRegex(DataError, "short/1"):
            analyze(self.root)


if __name__ == "__main__":
    unittest.main()
