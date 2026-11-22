"""Regenerate the portfolio figure from observed.json (requires matplotlib)."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "observed.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7))
    colors = ["#975022", "#097764", "#495e9a"]
    for ax, case in zip(axes, data["cases"]):
        samples = case["samples_ms_per_call"]
        for row in samples:
            ax.plot([0, 1, 2], [row["allocate"], row["reuse"], row["reuse_copy"]], color="#b6c4bf", linewidth=.8, alpha=.67)
        for x, key in enumerate(("allocate", "reuse", "reuse_copy")):
            values = [s[key] for s in samples]
            ax.scatter([x]*len(values), values, color=colors[x], s=10, alpha=.5, zorder=2)
            y = statistics.median(values)
            ax.scatter(x, y, s=78, marker="D", color=colors[x], edgecolor="white", linewidth=.9, zorder=3)
        ax.set_xticks([0, 1, 2], ["Fresh", "Reuse", "Reuse + copy"])
        ax.set_xlim(-.32, 2.32)
        ax.set_ylim(bottom=0)
        ax.set_title(f'{case["shape"][0]:,} rows × {case["shape"][1]:,} features', loc="left", fontsize=12, weight="bold")
        ax.set_ylabel("Elapsed time per call (ms)")
        ax.grid(axis="y", color="#dde5e1", linewidth=.8)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#a8b5b0")
        ax.set_axisbelow(True)
    fig.suptitle("CPU residual addition + RMSNorm", x=.08, y=.97, ha="left", weight="bold", fontsize=17)
    fig.text(.08,.885,"NumPy float32  ·  24 matched rounds per shape  ·  Independent vertical scales", fontsize=10, color="#456151")
    fig.text(.08,.055,"Each line connects three implementations from one round. Diamonds mark medians. Reuse + copy returns a fresh output.",fontsize=9, color="#456151")
    fig.subplots_adjust(left=.08,right=.97,bottom=.19,top=.75,wspace=.35)
    fig.savefig(root/"latency.png",dpi=175,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
