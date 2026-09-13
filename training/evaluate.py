"""Final Experiment 1 milestone check on the held-out test split:

Can the network recover amp parameters from unseen rendered audio, and
does the reconstructed audio actually sound like the original?

Usage (from repo root):
    python -m training.evaluate --config configs/experiment1.yaml --checkpoint best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from audio.features.spectrogram import LogMelFeature
from audio.preprocessing.io import load_wav
from audio.rendering.params import PARAM_NAMES, denormalize
from audio.rendering.virtual_amp import render as amp_render
from evaluation.metrics import parameter_mae, audio_similarity
from evaluation.visualizations import plot_waveform_and_spectrogram_comparison
from models.tone_predictor.model import TonePredictor
from training.dataset import ToneDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--n-plots", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = config["dataset"]["sr"]
    dataset_dir = REPO_ROOT / config["dataset"]["dataset_dir"]

    feature_fn = LogMelFeature(
        sr=sr,
        n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"],
        n_mels=config["features"]["n_mels"],
    )
    test_ds = ToneDataset(dataset_dir, "test", feature_fn, sr)

    checkpoint_path = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / args.checkpoint
    model = TonePredictor().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"] / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_example = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            feature, target, example_id = test_ds[idx]
            pred = model(feature.unsqueeze(0).to(device)).cpu().numpy()[0]
            true = target.numpy()

            example_dir = test_ds.example_dir(idx)
            source_audio, _ = load_wav(example_dir / "input.wav", sr=sr)
            ref_audio, _ = load_wav(example_dir / "audio.wav", sr=sr)

            pred_params = dict(zip(PARAM_NAMES, denormalize(pred.tolist())))
            gen_audio = amp_render(source_audio, sr, pred_params)

            mae = parameter_mae(pred, true)
            sim = audio_similarity(ref_audio, gen_audio)

            per_example.append({
                "id": example_id,
                "param_mae": mae,
                "audio_similarity": sim,
            })

            if idx < args.n_plots:
                plot_waveform_and_spectrogram_comparison(
                    ref_audio, gen_audio, sr,
                    run_dir / f"{example_id}_comparison.png",
                    title=f"example {example_id}",
                )

    overall = {
        "param_mae_mean": float(np.mean([e["param_mae"]["mean"] for e in per_example])),
        "audio_similarity_mean": float(np.mean([e["audio_similarity"]["similarity"] for e in per_example])),
        "spectral_convergence_mean": float(np.mean([e["audio_similarity"]["spectral_convergence"] for e in per_example])),
        "n_test_examples": len(per_example),
    }
    for name in PARAM_NAMES:
        overall[f"param_mae_{name}"] = float(np.mean([e["param_mae"][name] for e in per_example]))

    report = {"overall": overall, "per_example": per_example}
    with open(run_dir / "test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(overall, indent=2))
    print(f"full report and comparison plots written to {run_dir}")


if __name__ == "__main__":
    main()
