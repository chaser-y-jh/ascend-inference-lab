"""Measure two NumPy implementations of residual addition followed by RMSNorm."""

import argparse
import json
import itertools
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np


EPS = 1e-6
SEED = 41
HIDDEN = 1024


def allocate_each_call(x, residual, gamma):
    z = x + residual
    rms = np.sqrt((z * z).mean(axis=1, keepdims=True) + EPS)
    return (z / rms) * gamma


class ReusedBuffers:
    def __init__(self, shape):
        self.z = np.empty(shape, dtype=np.float32)
        self.output = np.empty(shape, dtype=np.float32)
        self.rms = np.empty((shape[0], 1), dtype=np.float32)

    def __call__(self, x, residual, gamma):
        np.add(x, residual, out=self.z)
        np.multiply(self.z, self.z, out=self.output)
        np.mean(self.output, axis=1, keepdims=True, out=self.rms)
        np.add(self.rms, EPS, out=self.rms)
        np.sqrt(self.rms, out=self.rms)
        np.divide(self.z, self.rms, out=self.output)
        np.multiply(self.output, gamma, out=self.output)
        return self.output  # The next call reuses and overwrites this result.


def elapsed_ms(fn, inputs, repetitions):
    start = time.perf_counter_ns()
    for _ in range(repetitions):
        fn(*inputs)
    return (time.perf_counter_ns() - start) / repetitions / 1_000_000


def run_case(rows, repetitions, pairs):
    rng = np.random.default_rng(SEED + rows)
    shape = (rows, HIDDEN)
    x = rng.standard_normal(shape, dtype=np.float32)
    residual = rng.standard_normal(shape, dtype=np.float32)
    gamma = rng.standard_normal(HIDDEN, dtype=np.float32)
    inputs = (x, residual, gamma)
    z64 = x.astype(np.float64) + residual.astype(np.float64)
    reference = z64 / np.sqrt(np.mean(z64**2, axis=1, keepdims=True) + EPS) * gamma
    reuse_owned = ReusedBuffers(shape)
    def copy_output(*args):
        return reuse_owned(*args).copy()

    fns = {"allocate": allocate_each_call, "reuse": ReusedBuffers(shape),
           "reuse_copy": copy_output}
    errors = {}
    for name, fn in fns.items():
        value = fn(*inputs)
        np.testing.assert_allclose(value, reference, rtol=2e-5, atol=2e-6)
        errors[name] = float(np.max(np.abs(value - reference)))
    del reference, z64, value
    for _ in range(5):
        for fn in fns.values():
            fn(*inputs)

    permutations = list(itertools.permutations(fns))
    if pairs % len(permutations):
        raise ValueError("The observation count must balance the six run orders.")
    orders = permutations * (pairs // len(permutations))
    random.Random(SEED + rows).shuffle(orders)
    samples = []
    for order in orders:
        observation = {"order": list(order)}
        for name in order:
            observation[name] = elapsed_ms(fns[name], inputs, repetitions)
        samples.append(observation)
    medians = {name: statistics.median(p[name] for p in samples) for name in fns}
    paired_deltas = [p["allocate"] - p["reuse"] for p in samples]
    quartiles = statistics.quantiles(paired_deltas, n=4, method="inclusive")
    return {"shape": list(shape), "calls_per_observation": repetitions, "pairs": pairs,
            "max_abs_error_vs_float64": errors, "median_ms": medians,
            "paired_delta_ms": {"q1": quartiles[0], "median": statistics.median(paired_deltas),
                                "q3": quartiles[2]},
            "samples_ms_per_call": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"device": "CPU", "backend": "NumPy", "dtype": "float32", "eps": EPS,
              "python": platform.python_version(), "numpy": np.__version__,
              "os": platform.system(), "architecture": platform.machine(),
              "processor_family": platform.processor(), "seed": SEED,
              "timing": "wall-clock ms/call, fixed inputs; reuse and reuse_copy preallocate scratch outside timer; reuse_copy includes a fresh output copy",
              "tolerance": {"rtol": 2e-5, "atol": 2e-6},
              "cases": [run_case(128, 10, 24), run_case(2048, 3, 24)]}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for case in report["cases"]:
        print(case["shape"], case["median_ms"], case["paired_delta_ms"])


if __name__ == "__main__":
    main()
