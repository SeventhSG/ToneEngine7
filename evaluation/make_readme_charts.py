"""Render the README's result charts from real experiment output.

Every number plotted here comes from training/experiments/<run>/history.json
and .../eval/test_report.json, produced by train.py and evaluate.py. Nothing
in this file invents data. Run training + evaluate first.

Usage (from repo root):
    python -m evaluation.make_readme_charts --run exp1_smoke
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
from audio.rendering.params import PARAM_NAMES, denormalize
from audio.rendering.virtual_amp import render as amp_render
from models.tone_predictor.model import TonePredictor
from training.dataset import ToneDataset

REPO_ROOT = Path(__file__).resolve().parents[1]

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "grid": "#e4e3de",
        "series_1": "#2a78d6",
        "series_2": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "grid": "#333230",
        "series_1": "#3987e5",
        "series_2": "#d95926",
    },
}


def style_axes(fig, ax, theme):
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(theme["grid"])
    ax.tick_params(colors=theme["text_secondary"], labelsize=9)
    ax.xaxis.label.set_color(theme["text_secondary"])
    ax.yaxis.label.set_color(theme["text_secondary"])
    ax.title.set_color(theme["text_primary"])
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8, zorder=0)


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
    ax.set_ylabel("MSE (normalized parameters)")
    ax.set_title("Experiment 1: training curve")
    legend = ax.legend(frameon=False, labelcolor=theme["text_primary"], loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"loss_curve_{theme_name}.png", facecolor=theme["surface"])
    plt.close(fig)


def plot_param_mae(test_report, out_dir, theme_name):
    theme = THEMES[theme_name]
    overall = test_report["overall"]
    values = [overall[f"param_mae_{name}"] for name in PARAM_NAMES]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    style_axes(fig, ax, theme)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    y_pos = np.arange(len(PARAM_NAMES))
    bars = ax.barh(y_pos, values, color=theme["series_1"], height=0.55, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(PARAM_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel("mean absolute error (0-10 knob scale)")
    ax.set_xlim(0, 10)
    ax.set_title("Experiment 1: parameter error on held-out test set")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}", va="center", ha="left",
            color=theme["text_primary"], fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_dir / f"param_mae_{theme_name}.png", facecolor=theme["surface"])
    plt.close(fig)


def plot_audio_similarity(history, out_dir, theme_name):
    theme = THEMES[theme_name]
    points = [h for h in history if "audio_similarity_mean" in h]
    epochs = [h["epoch"] for h in points]
    similarity = [h["audio_similarity_mean"] for h in points]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    style_axes(fig, ax, theme)
    ax.plot(epochs, similarity, color=theme["series_1"], linewidth=2, marker="o", markersize=5, zorder=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("audio similarity (validation)")
    ax.set_ylim(0, 1)
    ax.set_title("Experiment 1: reconstructed audio similarity during training")
    fig.tight_layout()
    fig.savefig(out_dir / f"audio_similarity_{theme_name}.png", facecolor=theme["surface"])
    plt.close(fig)


def plot_reconstruction_example(config_path, run_dir, out_dir, theme_name):
    theme = THEMES[theme_name]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sr = config["dataset"]["sr"]
    dataset_dir = REPO_ROOT / config["dataset"]["dataset_dir"]
    feature_fn = LogMelFeature(
        sr=sr, n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"], n_mels=config["features"]["n_mels"],
    )
    test_ds = ToneDataset(dataset_dir, "test", feature_fn, sr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TonePredictor().to(device)
    checkpoint = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    with open(run_dir / "eval" / "test_report.json") as f:
        report = json.load(f)
    per_example = sorted(report["per_example"], key=lambda e: e["audio_similarity"]["similarity"])
    median_id = per_example[len(per_example) // 2]["id"]
    idx = next(i for i in range(len(test_ds)) if test_ds.rows[i]["id"] == median_id)

    with torch.no_grad():
        feature, _, example_id = test_ds[idx]
        pred = model(feature.unsqueeze(0).to(device)).cpu().numpy()[0]

    example_dir = test_ds.example_dir(idx)
    source_audio, _ = load_wav(example_dir / "input.wav", sr=sr)
    ref_audio, _ = load_wav(example_dir / "audio.wav", sr=sr)
    pred_params = dict(zip(PARAM_NAMES, denormalize(pred.tolist())))
    gen_audio = amp_render(source_audio, sr, pred_params)

    n_fft = 512
    hop = n_fft // 4
    window = torch.hann_window(n_fft)
    ref_spec = torch.stft(torch.as_tensor(ref_audio), n_fft, hop, window=window, return_complex=True)
    gen_spec = torch.stft(torch.as_tensor(gen_audio), n_fft, hop, window=window, return_complex=True)
    ref_db = 20 * np.log10(np.abs(ref_spec.numpy()) + 1e-6)
    gen_db = 20 * np.log10(np.abs(gen_spec.numpy()) + 1e-6)
    vmin, vmax = ref_db.min(), ref_db.max()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=160)
    fig.patch.set_facecolor(theme["surface"])
    cmap = "magma" if theme_name == "light" else "inferno"

    for ax, spec, title in zip(axes, [ref_db, gen_db], ["reference (median test example)", "reconstructed from predicted parameters"]):
        ax.set_facecolor(theme["surface"])
        ax.imshow(spec, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color=theme["text_primary"], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Experiment 1: reference vs. reconstructed tone", color=theme["text_primary"], fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"reconstruction_example_{theme_name}.png", facecolor=theme["surface"])
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

    out_dir = REPO_ROOT / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else REPO_ROOT / "configs" / f"{args.run}.yaml"
    if not config_path.exists():
        config_path = REPO_ROOT / "configs" / "experiment1.yaml"

    for theme_name in THEMES:
        plot_loss_curve(history, out_dir, theme_name)
        plot_param_mae(test_report, out_dir, theme_name)
        plot_audio_similarity(history, out_dir, theme_name)
        plot_reconstruction_example(config_path, run_dir, out_dir, theme_name)

    print(f"wrote charts to {out_dir}")


if __name__ == "__main__":
    main()
