"""Experiment 2 milestone check: can the network pick the right amp, the
right cabinet, whether an overdrive pedal was used, and the right knob
values, from unseen rendered audio?

Usage (from repo root):
    python -m training.evaluate_exp2 --config configs/experiment2.yaml --checkpoint best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from audio.features.spectrogram import LogMelFeature
from audio.preprocessing.io import load_wav
from audio.rendering.chain_params import CONTINUOUS_NAMES, PARAM_MIN, PARAM_MAX
from audio.rendering.signal_chain import render as chain_render
from audio.similarity.stft_loss import multi_resolution_stft_distance
from evaluation.visualizations import plot_waveform_and_spectrogram_comparison
from models.tone_predictor.chain_model import ChainPredictor
from training.dataset_exp2 import ChainDataset
from training.train_exp2 import pred_to_chain, _OVERDRIVE_MASK_IDX

REPO_ROOT = Path(__file__).resolve().parents[1]


def continuous_mae(pred_vector01, target, row):
    scale = PARAM_MAX - PARAM_MIN
    true_vector01 = target["continuous"].numpy()
    abs_err = np.abs(pred_vector01 - true_vector01) * scale
    mask = np.ones_like(abs_err)
    if target["overdrive_on"].item() < 0.5:
        mask[_OVERDRIVE_MASK_IDX] = 0.0
    if mask.sum() == 0:
        return 0.0
    return float((abs_err * mask).sum() / mask.sum())


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
        sr=sr, n_fft=config["features"]["n_fft"],
        hop_length=config["features"]["hop_length"], n_mels=config["features"]["n_mels"],
    )
    test_ds = ChainDataset(dataset_dir, "test", feature_fn, sr)

    checkpoint_path = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / args.checkpoint
    model = ChainPredictor().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"] / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    per_example = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            feature, target, example_id = test_ds[idx]
            pred = model(feature.unsqueeze(0).to(device))

            example_dir = test_ds.example_dir(idx)
            source_audio, _ = load_wav(example_dir / "input.wav", sr=sr)
            ref_audio, _ = load_wav(example_dir / "audio.wav", sr=sr)

            chain = pred_to_chain(pred, 0)
            gen_audio = chain_render(source_audio, sr, chain)

            pred_vector01 = pred["continuous"][0].cpu().numpy()
            mae = continuous_mae(pred_vector01, target, None)
            sim = multi_resolution_stft_distance(ref_audio, gen_audio)

            amp_correct = int(pred["amp_logits"][0].argmax().item() == target["amp_id"].item())
            cabinet_correct = int(pred["cabinet_logits"][0].argmax().item() == target["cabinet_id"].item())
            overdrive_pred = torch.sigmoid(pred["overdrive_logit"][0]).item() > 0.5
            overdrive_correct = int(overdrive_pred == bool(target["overdrive_on"].item() > 0.5))

            per_example.append({
                "id": example_id,
                "continuous_mae": mae,
                "amp_correct": amp_correct,
                "cabinet_correct": cabinet_correct,
                "overdrive_correct": overdrive_correct,
                "audio_similarity": sim["similarity"],
                "spectral_convergence": sim["spectral_convergence"],
            })

            if idx < args.n_plots:
                plot_waveform_and_spectrogram_comparison(
                    ref_audio, gen_audio, sr,
                    run_dir / f"{example_id}_comparison.png",
                    title=f"example {example_id}",
                )

    overall = {
        "n_test_examples": len(per_example),
        "continuous_mae_mean": float(np.mean([e["continuous_mae"] for e in per_example])),
        "audio_similarity_mean": float(np.mean([e["audio_similarity"] for e in per_example])),
        "spectral_convergence_mean": float(np.mean([e["spectral_convergence"] for e in per_example])),
        "amp_accuracy": float(np.mean([e["amp_correct"] for e in per_example])),
        "cabinet_accuracy": float(np.mean([e["cabinet_correct"] for e in per_example])),
        "overdrive_accuracy": float(np.mean([e["overdrive_correct"] for e in per_example])),
    }

    report = {"overall": overall, "per_example": per_example}
    with open(run_dir / "test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(overall, indent=2))
    print(f"full report and comparison plots written to {run_dir}")


if __name__ == "__main__":
    main()
