"""Does the optimization loop actually help?

For each test example: take the trained CNN's prediction (already measured
in training.evaluate), then refine it with optimizer.optimize and re-measure.
Reports parameter MAE and audio similarity before vs. after, on the same
metric used throughout the project, so the comparison is apples to apples.

Usage (from repo root):
    python -m optimization.evaluate_optimization --config configs/experiment1.yaml --n-examples 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from audio.features.spectrogram import LogMelFeature
from audio.preprocessing.io import load_wav
from audio.rendering.params import PARAM_NAMES, denormalize
from audio.rendering.virtual_amp import render as amp_render
from evaluation.metrics import parameter_mae, audio_similarity
from models.tone_predictor.model import TonePredictor
from optimization.optimizer import optimize
from training.dataset import ToneDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--max-evals", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
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
    test_ds = ToneDataset(dataset_dir, "test", feature_fn, sr)
    n_examples = min(args.n_examples, len(test_ds)) if args.n_examples > 0 else len(test_ds)

    checkpoint_path = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / args.checkpoint
    model = TonePredictor().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    per_example = []
    t0 = time.time()
    with torch.no_grad():
        for idx in range(n_examples):
            feature, target, example_id = test_ds[idx]
            pred01 = model(feature.unsqueeze(0).to(device)).cpu().numpy()[0]
            true01 = target.numpy()

            example_dir = test_ds.example_dir(idx)
            source_audio, _ = load_wav(example_dir / "input.wav", sr=sr)
            ref_audio, _ = load_wav(example_dir / "audio.wav", sr=sr)

            before_params = dict(zip(PARAM_NAMES, denormalize(pred01.tolist())))
            before_audio = amp_render(source_audio, sr, before_params)
            before_mae = parameter_mae(pred01, true01)
            before_sim = audio_similarity(ref_audio, before_audio)

            optimized01, n_evals = optimize(
                pred01, source_audio, sr, ref_audio,
                max_evals=args.max_evals, seed=args.seed,
            )
            after_params = dict(zip(PARAM_NAMES, denormalize(optimized01.tolist())))
            after_audio = amp_render(source_audio, sr, after_params)
            after_mae = parameter_mae(optimized01, true01)
            after_sim = audio_similarity(ref_audio, after_audio)

            per_example.append({
                "id": example_id,
                "n_evals": n_evals,
                "before": {"param_mae": before_mae["mean"], "similarity": before_sim["similarity"]},
                "after": {"param_mae": after_mae["mean"], "similarity": after_sim["similarity"]},
            })

    elapsed = time.time() - t0

    overall = {
        "n_examples": n_examples,
        "max_evals": args.max_evals,
        "seconds": elapsed,
        "param_mae_before_mean": float(np.mean([e["before"]["param_mae"] for e in per_example])),
        "param_mae_after_mean": float(np.mean([e["after"]["param_mae"] for e in per_example])),
        "similarity_before_mean": float(np.mean([e["before"]["similarity"] for e in per_example])),
        "similarity_after_mean": float(np.mean([e["after"]["similarity"] for e in per_example])),
        "similarity_improved_fraction": float(np.mean([
            e["after"]["similarity"] > e["before"]["similarity"] for e in per_example
        ])),
    }

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"] / "optimization"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "report.json", "w") as f:
        json.dump({"overall": overall, "per_example": per_example}, f, indent=2)

    print(json.dumps(overall, indent=2))
    print(f"report written to {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
