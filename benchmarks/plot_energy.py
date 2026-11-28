"""Integrate matched power captures and plot energy per unit of completed work."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_series(path, columns):
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    try:
        data = np.asarray([[float(row[column]) for column in columns] for row in rows])
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(f"{path} requires numeric columns {', '.join(columns)}") from error
    if len(data) < 2 or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"{path} requires at least two finite samples in increasing time order")
    return data.T


def measure(run, root, source, max_gap_s):
    label = run["label"]
    start, end = (float(value) for value in run["window_s"])
    units = float(run["units"])
    if not (np.isfinite([start, end, units]).all() and start < end and units > 0):
        raise ValueError(f"{label}: expected a positive work count and a valid time window")
    columns = {"meter": ("power_w",), "rail": ("energy_j",),
               "battery": ("current_ua", "voltage_uv", "powered")}[source]
    time, *values = read_series(root / run["csv"], ("time_s", *columns))
    if not (time[0] <= start < end <= time[-1]):
        raise ValueError(f"{label}: power samples do not bracket the measurement window")
    lo = np.searchsorted(time, start, side="right") - 1
    hi = np.searchsorted(time, end, side="left") + 1
    time = time[lo:hi]
    values = [column[lo:hi] for column in values]
    if np.any(np.diff(time) > max_gap_s):
        raise ValueError(f"{label}: gap exceeds the campaign's max_gap_s")
    if source == "battery":
        current, voltage, powered = values
        if np.any(powered != 0) or np.any(current > 0) or np.any(voltage <= 0):
            raise ValueError(f"{label}: battery energy requires unplugged, discharging samples")
        samples = -current * voltage / 1e12
    else:
        samples, = values
        if np.any(samples < 0) or (source == "rail" and np.any(np.diff(samples) < 0)):
            raise ValueError(f"{label}: invalid power values or a reset cumulative rail counter")
    clipped = np.r_[start, time[(time > start) & (time < end)], end]
    progress = None
    if "progress_csv" in run:
        progress_time, count = read_series(root / run["progress_csv"], ("time_s", "completed_units"))
        if progress_time[0] > start or np.any(count < 0) or np.any(np.diff(count) < 0):
            raise ValueError(f"{label}: progress needs an initial count and nondecreasing work")
        initial = count[np.searchsorted(progress_time, start, side="right") - 1]
        final = count[np.searchsorted(progress_time, end, side="right") - 1]
        if not np.isclose(final - initial, units):
            raise ValueError(f"{label}: progress event counts disagree with declared work")
        events = progress_time[(progress_time > start) & (progress_time < end)]
        clipped = np.union1d(clipped, events)
        progress = count[np.searchsorted(progress_time, clipped, side="right") - 1] - initial
    samples = np.interp(clipped, time, samples)
    deltas = np.diff(clipped)
    if source == "rail":
        increments = np.diff(samples)
        power = np.r_[increments / deltas, increments[-1] / deltas[-1]]
    else:
        power = samples
        increments = (samples[:-1] + samples[1:]) * deltas / 2
    energy = np.r_[0.0, np.cumsum(increments)]
    total = energy[-1]
    if total <= 0:
        raise ValueError(f"{label}: observed zero energy in the measurement window")

    efficiency = None
    if progress is not None:
        consumed = energy[1:]
        before = np.divide(progress[:-1], consumed, out=np.full_like(consumed, np.nan), where=consumed > 0)
        after = np.divide(progress[1:], consumed, out=np.full_like(consumed, np.nan), where=consumed > 0)
        efficiency = (np.repeat(clipped[1:] - start, 2), np.column_stack((before, after)).ravel())

    elapsed = end - start
    summary = {"label": label, "units": units, "seconds": elapsed, "joules": float(total),
               "mean_watts": float(total / elapsed), "joules_per_unit": float(total / units),
               "units_per_joule": float(units / total), "units_per_second": units / elapsed}
    return summary, clipped - start, power, energy, efficiency


def plot(campaign, measured, output):
    unit = campaign["work_unit"]
    title = campaign["title"]
    scope = campaign["scope"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2), layout="constrained")
    figure.suptitle(f"{title} · {scope}")
    summaries = []
    for index, (summary, time, power, energy, efficiency) in enumerate(measured):
        label = summary["label"]
        color = f"C{index % 10}"
        style = {"drawstyle": "steps-post"} if campaign["source"] == "rail" else {}
        axes[0, 0].plot(time, power, color=color, label=label, **style)
        axes[0, 1].plot(time, energy, color=color, label=label)
        bars = axes[1, 0].bar(index, summary["units_per_joule"], color=color)
        axes[1, 0].bar_label(bars, fmt="%.3g", padding=3)
        if efficiency is not None:
            axes[1, 1].plot(*efficiency, color=color, label=label)
        summaries.append(summary)
    axes[0, 0].set(title="Power over time", xlabel="Elapsed time (s)", ylabel="W")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title="Cumulative energy", xlabel="Elapsed time (s)", ylabel="J")
    axes[1, 0].set(title="Total energy efficiency", ylabel=f"{unit}/J", xticks=range(len(summaries)),
                   xticklabels=[item["label"] for item in summaries])
    axes[1, 0].margins(y=0.15)
    if any(record[-1] is not None for record in measured):
        axes[1, 1].set(title="Cumulative energy efficiency", xlabel="Elapsed time (s)", ylabel=f"{unit}/J")
    else:
        for index, item in enumerate(summaries):
            axes[1, 1].scatter(item["mean_watts"], item["units_per_second"], color=f"C{index % 10}")
            axes[1, 1].annotate(item["label"], (item["mean_watts"], item["units_per_second"]),
                                xytext=(4, 4), textcoords="offset points")
        axes[1, 1].set(title="Throughput and mean power", xlabel="Average W", ylabel=f"{unit}/s")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "energy-comparison.png", dpi=160, facecolor="white")
    plt.close(figure)
    data = {"title": title, "scope": scope, "source": campaign["source"], "work_unit": unit, "runs": summaries}
    (output / "energy-summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    gap = float(campaign.get("max_gap_s", 2))
    if not np.isfinite(gap) or gap <= 0 or campaign["source"] not in ("meter", "rail", "battery"):
        parser.error("Use meter, rail, or battery and a positive max_gap_s")
    if not campaign["runs"] or not campaign["scope"] or not campaign["work_unit"]:
        parser.error("Declare measurement scope, work unit, and at least one run")
    try:
        measured = [measure(run, args.campaign.parent, campaign["source"], gap) for run in campaign["runs"]]
    except ValueError as error:
        parser.error(str(error))
    plot(campaign, measured, args.out)


if __name__ == "__main__":
    main()
