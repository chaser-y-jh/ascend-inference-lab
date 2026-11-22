"""Compare repeated Qwen3-8B Dense RMSNorm trials from CANN Learning Hub."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


class DataError(ValueError):
    """One or more runs cannot be compared using this experiment design."""


def read_json(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise DataError(f"{path.name}: expected a JSON object")
    return data


def required(condition, message):
    if not condition:
        raise DataError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(folder, name, fused, used):
    path = folder / name
    digest = sha256(path)
    required(digest not in used, "reused timing summary; each repeat needs its own run")
    used.add(digest)
    run = read_json(path)
    required(run["profiler_enabled"] is False, "Profiler must be off for timing")
    flags = run["rmsnorm_status"]
    required(all(flags[k] is fused for k in ("rmsnorm_fused", "add_rmsnorm_fused")),
             f"{name}: incorrect RMSNorm switch combination")
    required(run["batch_size"] == run["num_prompts"] == 1, "requires one prompt and batch=1")
    required(run["dtype"] in ("bfloat16", "bf16") and run["recipe_exe_mode"] == "eager",
             "expected BF16 eager execution")
    seconds = [run[k] for k in ("prefill_time_s", "decode_time_s")]
    required(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in seconds),
             f"{name}: missing or invalid stage timing; cannot use wall time")
    elapsed = sum(seconds)
    required(math.isclose(run["total_time_s"], elapsed, rel_tol=1e-8),
             f"{name}: total_time_s differs from prefill + decode")
    for k in ("input_tokens", "generated_tokens", "max_input_tokens", "max_new_tokens"):
        required(type(run[k]) is int and run[k] > 0, f"{name}: invalid {k}")
    required(math.isclose(run["tokens_per_second"], run["generated_tokens"] / elapsed,
                          rel_tol=1e-6), f"{name}: inconsistent token rate")
    conditions = {k: run[k] for k in ("model_path", "device", "dtype", "batch_size",
                  "seed", "enable_thinking", "recipe_exe_mode", "max_input_tokens", "max_new_tokens",
                  "resolved_attn_implementation")}
    sample = {"summary_sha256": digest, "prefill_s": seconds[0], "decode_s": seconds[1],
              "log_stage_total_s": elapsed, "tokens_per_second": run["tokens_per_second"],
              "generated_tokens": run["generated_tokens"], "input_tokens": run["input_tokens"],
              "text_sanity": run["generation_text_sanity"]["status"]}
    return conditions, sample


def read_pair(folder, used):
    a_config, a = measure(folder, "baseline.json", False, used)
    b_config, b = measure(folder, "fused.json", True, used)
    generations = [read_json(folder / f"{name}.jsonl") for name in ("baseline", "fused")]
    workload = generations[0]["prompt"]
    required(isinstance(workload, str) and workload.strip() and workload == generations[1]["prompt"],
             "generation files must contain the same nonempty prompt")
    for generated, timing in zip(generations, (a, b)):
        required(generated["generated_tokens"] == timing["generated_tokens"],
                 "generation tokens do not match timing summary")
        output = generated["generated_text"]
        required(isinstance(output, str) and output.strip(), "missing output text")
        timing["output_text_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    required(a_config == b_config, "A/B configuration mismatch")
    required((a["input_tokens"], a["generated_tokens"]) == (b["input_tokens"], b["generated_tokens"]),
             "different actual token counts across A/B; use an equal-length pair")
    comparison = read_json(folder / "comparison.json")
    required(comparison["status"] == "ok", "A/B comparison did not finish")
    for key, sample in (("baseline_tokens_per_second", a), ("optimized_tokens_per_second", b)):
        required(math.isclose(comparison[key], sample["tokens_per_second"], rel_tol=1e-6),
                 "quality comparison belongs to a different timing run")
    checks = {key: comparison[key] for key in ("precision_check_ok", "precision_check_has_warning",
              "precision_check_hard_failed", "baseline_optimized_outputs_consistent")}
    required(all(type(v) is bool for v in checks.values()), "invalid quality flag")
    similarity = comparison["baseline_optimized_output_similarity_ratio"]
    required(type(similarity) in (int, float) and 0 <= similarity <= 1, "invalid output similarity")
    passed = (checks["precision_check_ok"] and not checks["precision_check_has_warning"]
              and not checks["precision_check_hard_failed"] and checks["baseline_optimized_outputs_consistent"]
              and a["text_sanity"] == b["text_sanity"] == "ok" and similarity >= 0.95)
    return a_config, {"repeat": folder.name, "prompt_sha256": hashlib.sha256(workload.encode()).hexdigest(),
            "baseline": a, "fused": b, "comparison_sha256": sha256(folder / "comparison.json"),
            "quality": {**checks, "text_similarity": similarity, "pass": passed}}


def spread(samples, mode):
    seconds = [sample[mode]["log_stage_total_s"] for sample in samples]
    return {"median_s": statistics.median(seconds), "min_s": min(seconds), "max_s": max(seconds)}


def analyze(root):
    environment = read_json(root / "environment.json")
    for key in ("device_model", "driver", "cann", "torch", "torch_npu", "weights_revision",
                "tutorial_commit", "decoding_policy", "warmup_policy"):
        required(isinstance(environment.get(key), str) and environment[key].strip(),
                 f"environment.json: missing {key}")
    results, used, counts = {}, set(), {}
    fixed = None
    for load in ("short", "long"):
        samples = []
        load_key = None
        for folder in sorted((root / load).iterdir()):
            if not folder.is_dir():
                continue
            try:
                config, row = read_pair(folder, used)
                if fixed is None:
                    fixed = config
                required(config == fixed, "experiment configuration changed across runs")
                current = (row["prompt_sha256"], row["baseline"]["input_tokens"], row["baseline"]["generated_tokens"])
                required(load_key is None or load_key == current, "prompt or actual token counts changed within workload")
                load_key = current
                samples.append(row)
            except (KeyError, ValueError, TypeError, OSError) as exc:
                raise DataError(f"{load}/{folder.name}: {exc}") from exc
        required(samples, f"{load}: no paired runs")
        counts[load] = load_key
        a, b = spread(samples, "baseline"), spread(samples, "fused")
        ready = all(row["quality"]["pass"] for row in samples) and len(samples) >= 3
        results[load] = {"repeats": len(samples), "baseline": a, "fused": b,
                         "quality_and_repeat_gate_passed": ready,
                         "median_time_change_pct": (b["median_s"] / a["median_s"] - 1) * 100 if ready else None,
                         "samples": samples}
    short, long = counts["short"], counts["long"]
    required(long[0] != short[0] and long[1] > short[1],
             "long input needs a different prompt and more actual input tokens")
    required(long[2] == short[2], "use the same actual generated token count across workloads")
    return {"metric": "logged prefill + decode time in seconds", "environment": environment,
            "fixed_conditions": fixed, "workloads": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="directory containing environment.json, short/ and long/")
    args = parser.parse_args()
    try:
        report = analyze(args.runs)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"Input error: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
