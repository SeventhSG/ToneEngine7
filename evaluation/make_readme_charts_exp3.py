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


def load_choice_search_reports(opt_dir):
    """The three 1,800-render runs the choice-search chart compares, restricted
    to the test examples every one of them covers (they all run n=200, indexed
    from 0, so the sets are identical, but this stays honest if that changes).
    Returns None if any run is missing."""
    names = {"fixed_1800": "report_fixed_b1800.json", "search_1800": "report_search_b1800.json",
             "oracle_1800": "report_oracle_b1800.json"}
    paths = {key: opt_dir / name for key, name in names.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    per_example = {}
    for key, path in paths.items():
        with open(path) as f:
            per_example[key] = {e["index"]: e for e in json.load(f)["per_example"]}
    shared = sorted(set.intersection(*(set(v) for v in per_example.values())))
    return {key: [v[i] for i in shared] for key, v in per_example.items()}


def plot_choice_search(reports, out_dir, theme_name):
    theme = THEMES[theme_name]
    n = len(reports["fixed_1800"])

    def mean(key, fn):
        return float(np.mean([fn(e) for e in reports[key]]))

    keys = ["fixed_1800", "search_1800", "oracle_1800"]
    labels = ["choices fixed\n1800 renders", "amp + comp\nsearch, 1800 renders",
              "true choices\n1800 renders\n(ceiling)"]
    colors = [theme["series_1"], theme["series_2"], None]
    cnn_similarity = mean("fixed_1800", lambda e: e["before"]["similarity"])

    panels = [
        ("audio similarity after optimization", lambda e: e["after"]["similarity"],
         cnn_similarity, f"CNN prediction only, before any optimization ({cnn_similarity:.2f})",
         lambda v: f"{v:.2f}"),
        ("amp identified correctly", lambda e: e["choices_correct"]["amp"],
         1 / len(SPEC.choices["amp"]), f"chance, {len(SPEC.choices['amp'])} amps "
         f"({1 / len(SPEC.choices['amp']):.0%})", lambda v: f"{v:.1%}".replace(".0%", "%")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=160)
    for ax, (title, fn, ref_value, ref_label, fmt) in zip(axes, panels):
        style_axes(fig, ax, theme)
        ax.grid(axis="x", visible=False)
        values = [mean(k, fn) for k in keys]
        x = np.arange(len(keys))
        bars = []
        for xi, value, color in zip(x, values, colors):
            if color is None:
                bars += ax.bar(xi, value, width=0.6, color=theme["surface"], edgecolor=theme["text_secondary"],
                               hatch="///", linewidth=1.2, zorder=3)
            else:
                bars += ax.bar(xi, value, width=0.6, color=color, zorder=3)
        ax.axhline(ref_value, color=theme["text_secondary"], linewidth=1.2, linestyle="--", zorder=4,
                   label=ref_label)
        ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=theme["text_secondary"],
                  handlelength=2.2, borderaxespad=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.set_ylim(0, 1.25)
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_yticks(ticks)
        ax.set_yticklabels([fmt(t) for t in ticks])
        ax.set_title(title, fontsize=10)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, fmt(value),
                    ha="center", va="bottom", color=theme["text_primary"], fontsize=10, zorder=5)

    fig.suptitle(f"Experiment 3: searching amp and compressor on/off alongside the knobs (same {n} test examples in every bar)",
                 color=theme["text_primary"], fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_dir / f"exp3_choice_search_{theme_name}.png", facecolor=theme["surface"])
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

    choice_reports = load_choice_search_reports(REPO_ROOT / "training" / "experiments" / args.run / "optimization")
    for theme_name in THEMES:
        if choice_reports is not None:
            plot_choice_search(choice_reports, out_dir, theme_name)

    print(f"wrote exp3 charts to {out_dir}")
    if choice_reports is None:
        print("choice-search reports incomplete; skipped the choice-search chart")


if __name__ == "__main__":
    main()
