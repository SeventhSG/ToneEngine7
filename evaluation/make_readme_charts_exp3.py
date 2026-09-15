"""Render the README's Experiment 3 chart from a real test report.

Usage (from repo root):
    python -m evaluation.make_readme_charts_exp3 --run exp3_1m
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audio.rendering.signal_chain3 import EXP3_SPEC as SPEC
from evaluation.make_readme_charts import THEMES, style_axes

REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = {"gate_on": "gate\non/off", "comp_on": "comp\non/off", "overdrive_on": "drive\non/off",
          "eq_on": "EQ\non/off", "amp": "amp\n(3)", "cabinet": "cabinet\n(4)", "mic": "mic\n(3)"}
GROUPS = [("gate", ["gate_threshold"]), ("compressor", ["comp_sustain", "comp_level"]),
          ("overdrive", ["od_drive", "od_level"]), ("amp", ["gain", "bass", "mid", "treble", "presence", "master"]),
          ("EQ pedal", ["eq_100", "eq_400", "eq_800", "eq_1600", "eq_3200"]), ("mic", ["mic_position", "mic_distance"])]


def plot(overall, run, out_dir, theme_name):
    theme = THEMES[theme_name]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160, gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    style_axes(fig, ax, theme)
    ax.grid(axis="x", visible=False)
    names = SPEC.switch_names + SPEC.choice_names
    values = [overall["switch_accuracy"][s] for s in SPEC.switch_names] + [overall["choice_accuracy"][c] for c in SPEC.choice_names]
    chance = [0.5] * len(SPEC.switch_names) + [1 / len(SPEC.choices[c]) for c in SPEC.choice_names]
    x = np.arange(len(names))
    colors = [theme["series_1"]] * len(SPEC.switch_names) + [theme["series_2"]] * len(SPEC.choice_names)
    bars = ax.bar(x, values, color=colors, width=0.62, zorder=3)
    for xi, c in zip(x, chance):
        ax.plot([xi - 0.34, xi + 0.34], [c, c], color=theme["text_secondary"], linewidth=1.4, linestyle="--", zorder=4)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", va="bottom",
                color=theme["text_primary"], fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names], fontsize=7.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("pedals on/off (blue) and gear choices (orange); dashed = chance", fontsize=10)

    ax = axes[1]
    style_axes(fig, ax, theme)
    ax.grid(axis="x", visible=False)
    xs, labels, vals, group_ticks = [], [], [], []
    pos = 0.0
    for group, knobs in GROUPS:
        start = pos
        for k in knobs:
            xs.append(pos)
            labels.append(k.split("_", 1)[1] if "_" in k and group != "amp" else k)
            vals.append(overall["knob_mae_by_knob"][k])
            pos += 1
        group_ticks.append(((start + pos - 1) / 2, group))
        pos += 0.8
    ax.bar(xs, vals, color=theme["series_1"], width=0.7, zorder=3)
    ax.axhline(10 / 3, color=theme["text_secondary"], linewidth=1.2, linestyle="--", zorder=4)
    ax.text(xs[-1] + 0.5, 10 / 3 + 0.08, "random guess (3.33)", ha="right", va="bottom",
            color=theme["text_secondary"], fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    for gx, g in group_ticks:  # pedal names under the knob labels (x in data units, y in axes units)
        ax.text(gx, -0.30, g, ha="center", va="top", color=theme["text_primary"], fontsize=8.5,
                transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 3.8)
    ax.set_ylabel("knob error, 0-10 scale (lower is better)")
    ax.set_title("every knob, counted only when its pedal is on", fontsize=10)

    fig.suptitle(f"Experiment 3: reading the full Stage 2 rig from audio, CNN only ({run}, {overall['n_test_examples']} test examples)",
                 color=theme["text_primary"], fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(out_dir / f"exp3_readout_{theme_name}.png", facecolor=theme["surface"])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()
    with open(REPO_ROOT / "training" / "experiments" / args.run / "eval" / "test_report.json") as f:
        overall = json.load(f)["overall"]
    out_dir = REPO_ROOT / "docs" / "assets"
    for theme_name in THEMES:
        plot(overall, args.run, out_dir, theme_name)
    print(f"wrote exp3 charts to {out_dir}")


if __name__ == "__main__":
    main()
