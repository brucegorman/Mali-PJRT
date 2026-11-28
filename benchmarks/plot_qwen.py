"""Render performance figures from the Mali Q4 decode result format."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CPU, MALI = "#295686", "#cd6729"


def read_result(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8"))


def save(figure, output, name):
    figure.savefig(output / name, dpi=160, facecolor="white")
    plt.close(figure)


def decode(root, output):
    result = read_result(root, "shape-profile/long-aligned/compare.json")
    if result["shader_timestamps"] or not result["qualification"]["correctness_passed"]:
        raise ValueError("The decode chart requires the verified unprofiled comparison")
    figure, (timeline, cold) = plt.subplots(1, 2, figsize=(12, 4.5),
                                            gridspec_kw={"width_ratios": [2, 1]}, layout="constrained")
    figure.suptitle("Qwen2.5-0.5B Q4_0 decode on vivo Y36m · matched 96-token run")
    names = ["CPU", "Mali hybrid"]
    cold_times = []
    for backend, name, color in zip(("cpu", "mali"), names, (CPU, MALI), strict=True):
        stats = result[backend]
        samples = np.asarray(stats["decode_ms"][1:])
        tokens = np.arange(3, len(samples) + 3)
        p50 = stats["warm_decode_p50_ms"]
        p95 = result["qualification"][f"{backend}_warm_p95_ms"]
        timeline.plot(tokens, samples, color=color, alpha=0.65,
                      label=f"{name}  p50 {p50:.1f} ms · p95 {p95:.1f} ms")
        timeline.axhline(p50, color=color, linestyle="--", linewidth=1)
        cold_times.append((stats["prefill_first_token_ms"], stats["first_decode_ms"]))
    timeline.set(title="Warm decode latency", xlabel="Generated token index", ylabel="ms per token")
    timeline.legend(loc="center right", frameon=False)
    for offset, name, times, color in zip((-0.18, 0.18), names, cold_times, (CPU, MALI), strict=True):
        bars = cold.bar(np.arange(2) + offset, times, width=0.34, color=color, label=name)
        cold.bar_label(bars, fmt="%.0f", padding=3)
    cold.set(title="Cold path", ylabel="ms", xticks=[0, 1], xticklabels=["Prefill to\nfirst token", "First decode\ngap"])
    cold.legend(frameon=False)
    cold.margins(y=0.2)
    save(figure, output, "qwen-decode.png")


def kernels(root, output):
    with (root / "kernel.csv").open(newline="", encoding="utf-8") as source:
        records = [row for row in csv.DictReader(source) if row["tensor"].startswith("blk.")]
    variants = {"split_fp32": "Expand + GEMV", "fused_shared_fp32": "Fused shared",
                "fused_subgroup_fp32": "Fused subgroup"}
    tensors = list(dict.fromkeys(row["tensor"] for row in records))
    values = {(row["tensor"], row["variant"]): float(row["gpu_median_ms"]) for row in records}
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for offset, (variant, label), color in zip((-0.24, 0, 0.24), variants.items(),
                                                (CPU, "#7b8d60", MALI), strict=True):
        bars = axis.bar(np.arange(len(tensors)) + offset,
                        [values[tensor, variant] for tensor in tensors], 0.22, color=color, label=label)
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axis.set(title="Vulkan kernel time · Qwen weights resident on Mali",
             ylabel="Median GPU time (ms)", xticks=range(len(tensors)),
             xticklabels=[name.removeprefix("blk.0.").removesuffix(".weight") for name in tensors])
    axis.legend(frameon=False)
    axis.margins(y=0.19)
    save(figure, output, "qwen-kernels.png")


def layouts(root, output):
    groups = (
        ("Weight layout", "weight-layout", ("1-raw", "2-soa", "3-soa", "4-raw"), ("Raw", "Aligned weights")),
        ("Activation layout", "activation-layout", ("1-weights", "2-both", "3-both", "4-weights"),
         ("Aligned weights", "Aligned weights + Q8")),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, layout="constrained")
    figure.suptitle("Qwen Mali hybrid layout ablations · matched ABBA diagnostic runs")
    for axis, (title, directory, cases, labels) in zip(axes, groups, strict=True):
        runs = [read_result(root, f"{directory}/{case}/compare.json") for case in cases]
        if not all(run["shader_timestamps"] for run in runs):
            raise ValueError("The layout ablations require shader timestamps")
        for indices, index, color in (((0, 3), 0, CPU), ((1, 2), 1, MALI)):
            values = [runs[i]["mali"]["warm_decode_p50_ms"] for i in indices]
            bars = axis.bar(index, mean(values), color=color, alpha=0.85, width=0.57)
            axis.scatter([index - 0.08, index + 0.08], values, color="black", s=13, zorder=3)
            axis.bar_label(bars, fmt="%.1f", padding=12)
        axis.set(title=title, xticks=[0, 1], xticklabels=labels)
        axis.margins(y=0.20)
    axes[0].set_ylabel("Warm p50 (ms/token)")
    save(figure, output, "qwen-layouts.png")


def breakdown(root, output):
    result = read_result(root, "shape-profile/aligned/compare.json")
    if not result["shader_timestamps"]:
        raise ValueError("The projection breakdown requires shader timestamps")
    shapes = result["mali"]["projection_shapes"]
    gpu = np.array([shape["gpu_timestamp_sum_ms"] for shape in shapes]) / 1000
    submit = np.array([shape["submit_wait_sum_ms"] for shape in shapes]) / 1000
    total = np.array([shape["gpu_call_sum_ms"] for shape in shapes]) / 1000
    if np.any(gpu > submit) or np.any(submit > total):
        raise ValueError("Expected GPU time inside the submit/wait interval and caller interval")
    figure, axis = plt.subplots(figsize=(9, 3.9), layout="constrained")
    labels = [f'{shape["rows"]:,} × {shape["cols"]:,}\n{shape["calls"]:,} calls' for shape in shapes]
    index = np.arange(len(shapes))
    axis.barh(index, gpu, color=MALI, label="GPU timestamp")
    axis.barh(index, submit - gpu, left=gpu, color="#b99773", label="Other submit/wait time")
    axis.barh(index, total - submit, left=submit, color=CPU, label="Other caller time")
    axis.set(title="Qwen aligned Vulkan projections · profiled 24-token run", xlabel="Total seconds",
             yticks=index, yticklabels=labels)
    axis.invert_yaxis()
    axis.legend(frameon=False, loc="lower right")
    save(figure, output, "qwen-projections.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-results", type=Path, default=ROOT / "data" / "qwen-y36m")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.22})
    args.out.mkdir(parents=True, exist_ok=True)
    decode(args.qwen_results, args.out)
    kernels(args.qwen_results, args.out)
    layouts(args.qwen_results, args.out)
    breakdown(args.qwen_results, args.out)


if __name__ == "__main__":
    main()
