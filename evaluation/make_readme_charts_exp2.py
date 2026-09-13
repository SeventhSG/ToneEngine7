"""Render the README's Experiment 2 result charts from real experiment output.

Mirrors evaluation/make_readme_charts.py's style (same THEMES, same
style_axes) but Experiment 2's targets are a mix of continuous knobs and
categorical/binary choices, so the charts differ. Nothing here is hand-tuned.

Usage (from repo root):
    python -m evaluation.make_readme_charts_exp2 --run exp2_smoke
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from audio.features.spectrogram import LogMelFeature
from audio.preprocessing.io import load_wav
from audio.rendering.chain_params import CONTINUOUS_NAMES, denormalize_continuous
from audio.rendering.signal_chain import render as chain_render
from evaluation.make_readme_charts import THEMES, style_axes, _spec_db
from models.tone_predictor.chain_model import ChainPredictor
from optimization.optimizer import optimize_generic
from training.dataset_exp2 import ChainDataset
from training.train_exp2 import pred_to_chain

REPO_ROOT = Path(__file__).resolve().parents[1]


def plot_loss_curve(history, out_dir, theme_name):
    theme = THEMES[theme_name]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    style_axes(fig, ax, theme)
    ax.plot(epochs, train_loss, color=theme["series_1"], linewidth=2, label="train loss", zorder=3)
    ax.plot(epochs, val_loss, color=theme["series_2"], linewidth=2, label="validation loss", zorder=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("multi-task loss (continuous + overdrive + amp + cabinet)")
    ax.set_title("Experiment 2: training curve")
    ax.legend(frameon=False, labelcolor=theme["text_primary"], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"exp2_loss_curve_{theme_name}.png", facecolor=theme["surface"])
    plt.close(fig)


def plot_classification_accuracy(test_report, out_dir, theme_name):
    theme = THEMES[theme_name]
    overall = test_report["overall"]
    labels = ["overdrive on/off\n(2 classes)", "amp choice\n(3 classes)", "cabinet choice\n(4 classes)"]
    chance = [0.5, 1 / 3, 1 / 4]
    values = [overall["overdrive_accuracy"], overall["amp_accuracy"], overall["cabinet_accuracy"]]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    style_axes(fig, ax, theme)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=theme["series_1"], width=0.5, zorder=3)
    for i, c in enumerate(chance):
        ax.plot([x[i] - 0.3, x[i] + 0.3], [c, c], color=theme["text_secondary"], linewidth=1.5, linestyle="--", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("test-set accuracy")
    ax.set_title("Experiment 2: signal chain classification accuracy (dashed = chance level)", fontsize=11, pad=12)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{value:.2f}",
                ha="center", va="bottom", color=theme["text_primary"], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"exp2_classification_accuracy_{theme_name}.png", facecolor=theme["surface"])
    plt.close(fig)


def plot_optimization_comparison(opt_report, out_dir, theme_name):
    theme = THEMES[theme_name]
    overall = opt_report["overall"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=160)
    fig.patch.set_facecolor(theme["surface"])

    metrics = [
        (axes[0], "audio similarity (higher is better)",
         overall["similarity_before_mean"], overall["similarity_after_mean"], (0, 1)),
        (axes[1], "continuous knob MAE, 0-10 scale (lower is better)",
         overall["continuous_mae_before_mean"], overall["continuous_mae_after_mean"], None),
    ]

    for ax, title, before, after, ylim in metrics:
        style_axes(fig, ax, theme)
        ax.grid(axis="x", visible=False)
        bars = ax.bar(
            ["CNN prediction", "after optimization"], [before, after],
            color=[theme["series_1"], theme["series_2"]], width=0.5, zorder=3,
        )
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10)
        for bar, value in zip(bars, [before, after]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                    ha="center", va="bottom", color=theme["text_primary"], fontsize=10)

    fig.suptitle(
        f"Experiment 2: optimization loop impact, discrete choices held fixed (n={overall['n_examples']} test examples)",
        color=theme["text_primary"], fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"exp2_optimization_comparison_{theme_name}.png", facecolor=theme["surface"])
    plt.close(fig)


def plot_reconstruction_example(config_path, run_dir, out_dir, theme_name, seed=0):
    theme = THEMES[theme_name]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sr = config["dataset"]["sr"]
    dataset_dir = REPO_ROOT / config["dataset"]["dataset_dir"]
    feature_fn = LogMelFeature(
        sr=sr, n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"], n_mels=config["features"]["n_mels"],
    )
    test_ds = ChainDataset(dataset_dir, "test", feature_fn, sr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChainPredictor().to(device)
    checkpoint = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    with open(run_dir / "eval" / "test_report.json") as f:
        report = json.load(f)
    per_example = sorted(report["per_example"], key=lambda e: e["audio_similarity"])
    median_id = per_example[len(per_example) // 2]["id"]
    idx = next(i for i in range(len(test_ds)) if test_ds.rows[i]["id"] == median_id)

    with torch.no_grad():
        feature, _, example_id = test_ds[idx]
        pred = model(feature.unsqueeze(0).to(device))

    example_dir = test_ds.example_dir(idx)
    source_audio, _ = load_wav(example_dir / "input.wav", sr=sr)
    ref_audio, _ = load_wav(example_dir / "audio.wav", sr=sr)

    chain = pred_to_chain(pred, 0)
    cnn_audio = chain_render(source_audio, sr, chain)

    def loss_fn(vector01, chain=chain):
        candidate = dict(chain)
        candidate.update(dict(zip(CONTINUOUS_NAMES, denormalize_continuous(vector01.tolist()))))
        gen_audio = chain_render(source_audio, sr, candidate)
        from audio.similarity.stft_loss import multi_resolution_stft_distance
        result = multi_resolution_stft_distance(ref_audio, gen_audio)
        return result["spectral_convergence"] + result["log_mag_error"]

    initial01 = pred["continuous"][0].cpu().numpy()
    optimized01, _ = optimize_generic(initial01, loss_fn, seed=seed)
    optimized_chain = dict(chain)
    optimized_chain.update(dict(zip(CONTINUOUS_NAMES, denormalize_continuous(optimized01.tolist()))))
    optimized_audio = chain_render(source_audio, sr, optimized_chain)

    ref_db = _spec_db(ref_audio)
    cnn_db = _spec_db(cnn_audio)
    opt_db = _spec_db(optimized_audio)
    vmin, vmax = ref_db.min(), ref_db.max()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), dpi=160)
    fig.patch.set_facecolor(theme["surface"])
    cmap = "magma" if theme_name == "light" else "inferno"

    panels = [
        (ref_db, "reference (median test example)"),
        (cnn_db, "CNN prediction only"),
        (opt_db, "after optimization"),
    ]
    for ax, (spec, title) in zip(axes, panels):
        ax.set_facecolor(theme["surface"])
        ax.imshow(spec, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color=theme["text_primary"], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Experiment 2: reference vs. reconstructed signal chain", color=theme["text_primary"], fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"exp2_reconstruction_example_{theme_name}.png", facecolor=theme["surface"])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run_dir = REPO_ROOT / "training" / "experiments" / args.run
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    with open(run_dir / "eval" / "test_report.json") as f:
        test_report = json.load(f)

    opt_report_path = run_dir / "optimization" / "report.json"
    opt_report = None
    if opt_report_path.exists():
        with open(opt_report_path) as f:
            opt_report = json.load(f)

    out_dir = REPO_ROOT / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else REPO_ROOT / "configs" / "experiment2.yaml"

    for theme_name in THEMES:
        plot_loss_curve(history, out_dir, theme_name)
        plot_classification_accuracy(test_report, out_dir, theme_name)
        plot_reconstruction_example(config_path, run_dir, out_dir, theme_name)
        if opt_report is not None:
            plot_optimization_comparison(opt_report, out_dir, theme_name)

    print(f"wrote charts to {out_dir}")
    if opt_report is None:
        print("no optimization/report.json found; skipped the optimization comparison chart")


if __name__ == "__main__":
    main()
