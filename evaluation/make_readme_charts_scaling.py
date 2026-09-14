"""Render the README's Experiment 2 data-scaling chart from real test reports.

Each point is one training run on a different amount of pool data
(configs/experiment2_pool*.yaml), all evaluated on the same exp2_smoke test
set. Runs without a test report yet are skipped, not interpolated.

Usage (from repo root):
    python -m evaluation.make_readme_charts_scaling
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.make_readme_charts import THEMES, style_axes

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = [("exp2_pool8k", 8000), ("exp2_pool32k", 32000), ("exp2_pool100k", 100000), ("exp2_pool300k", 300000),
        ("exp2_pool1m", 1000000)]


def load_points():
    points = []
    for run, n_train in RUNS:
        path = REPO_ROOT / "training" / "experiments" / run / "eval" / "test_report.json"
        if path.exists():
            with open(path) as f:
                points.append((n_train, json.load(f)["overall"]))
    return points


def _label(n):
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k"


def plot_scaling(points, out_dir, theme_name):
    theme = THEMES[theme_name]
    xs = [n for n, _ in points]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160)
    ax = axes[0]
    style_axes(fig, ax, theme)
    series = [("amp choice (3 amps)", "amp_accuracy", theme["series_1"], 1 / 3, "amp chance 33%"),
              ("overdrive on/off", "overdrive_accuracy", theme["series_2"], 1 / 2, "overdrive chance 50%")]
    values = {key: [o[key] for _, o in points] for _, key, _, _, _ in series}
    for label, key, color, chance, chance_label in series:
        ys = values[key]
        other = values["overdrive_accuracy" if key == "amp_accuracy" else "amp_accuracy"]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, label=label, zorder=3)
        ax.axhline(chance, color=color, linewidth=1, linestyle=":", alpha=0.8, zorder=2)
        ax.text(xs[-1] * 1.5, chance + 0.008, chance_label, ha="right", va="bottom",
                color=theme["text_secondary"], fontsize=8)
        for x, y, y_other in zip(xs, ys, other):
            above = y >= y_other  # the higher of the two series labels above its point, the lower below
            ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, 9 if above else -10),
                        ha="center", va="bottom" if above else "top", color=theme["text_primary"], fontsize=8.5)
    ax.set_ylim(0.25, 1.0)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([f"{t:.0%}" for t in ax.get_yticks()])
    ax.set_ylabel("test-set accuracy")
    ax.set_title("CNN identifies the gear", fontsize=10)
    ax.legend(frameon=False, labelcolor=theme["text_primary"], loc="upper left", fontsize=9)

    ax = axes[1]
    style_axes(fig, ax, theme)
    ys = [o["continuous_mae_mean"] for _, o in points]
    ax.plot(xs, ys, color=theme["series_1"], linewidth=2, marker="o", markersize=7, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                    color=theme["text_primary"], fontsize=8.5)
    ax.set_ylim(0, max(ys) * 1.25)
    ax.set_ylabel("knob mean absolute error, 0-10 scale")
    ax.set_title("CNN predicts the knobs (lower is better)", fontsize=10)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([_label(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlim(xs[0] / 1.6, xs[-1] * 1.6)
        ax.set_xlabel("training examples (log scale)")

    fig.suptitle("Experiment 2: effect of training data, CNN prediction only (same 1000 test examples)",
                 color=theme["text_primary"], fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"exp2_data_scaling_{theme_name}.png", facecolor=theme["surface"])
    plt.close(fig)


def main():
    points = load_points()
    if len(points) < 2:
        print("need at least two finished runs with test reports")
        return
    out_dir = REPO_ROOT / "docs" / "assets"
    for theme_name in THEMES:
        plot_scaling(points, out_dir, theme_name)
    print(f"wrote scaling charts for {[_label(n) for n, _ in points]} to {out_dir}")


if __name__ == "__main__":
    main()
