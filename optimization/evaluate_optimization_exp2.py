"""Does the optimization loop help on Experiment 2's signal chain?

The discrete choices (overdrive on/off, amp, cabinet) are fixed at the
CNN's prediction; CMA-ES only refines the continuous knobs against the
fixed chain. Extending the search to discrete choices too (try the runner-up
amp, try flipping overdrive) is a natural next step, noted as a limitation.

Usage (from repo root):
    python -m optimization.evaluate_optimization_exp2 --config configs/experiment2.yaml --n-examples 200
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
from audio.rendering.chain_params import CONTINUOUS_NAMES, PARAM_MIN, PARAM_MAX, denormalize_continuous
from audio.rendering.signal_chain import render as chain_render
from audio.similarity.stft_loss import multi_resolution_stft_distance
from models.tone_predictor.chain_model import ChainPredictor
from optimization.optimizer import optimize_generic
from training.dataset_exp2 import ChainDataset
from training.evaluate_exp2 import continuous_mae
from training.train_exp2 import pred_to_chain

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
    test_ds = ChainDataset(dataset_dir, "test", feature_fn, sr)
    n_examples = min(args.n_examples, len(test_ds)) if args.n_examples > 0 else len(test_ds)

    checkpoint_path = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / args.checkpoint
    model = ChainPredictor().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"] / "optimization"
    run_dir.mkdir(parents=True, exist_ok=True)
    partial_path = run_dir / "report_partial.json"

    per_example = []
    t0 = time.time()
    with torch.no_grad():
        for idx in range(n_examples):
            feature, target, example_id = test_ds[idx]
            pred = model(feature.unsqueeze(0).to(device))

            example_dir = test_ds.example_dir(idx)
            source_audio, _ = load_wav(example_dir / "input.wav", sr=sr)
            ref_audio, _ = load_wav(example_dir / "audio.wav", sr=sr)

            chain = pred_to_chain(pred, 0)
            before_audio = chain_render(source_audio, sr, chain)
            before_mae = continuous_mae(pred["continuous"][0].cpu().numpy(), target, None)
            before_sim = multi_resolution_stft_distance(ref_audio, before_audio)["similarity"]

            def loss_fn(vector01, chain=chain):
                candidate = dict(chain)
                candidate.update(dict(zip(CONTINUOUS_NAMES, denormalize_continuous(vector01.tolist()))))
                gen_audio = chain_render(source_audio, sr, candidate)
                result = multi_resolution_stft_distance(ref_audio, gen_audio)
                return result["spectral_convergence"] + result["log_mag_error"]

            initial01 = pred["continuous"][0].cpu().numpy()
            optimized01, n_evals = optimize_generic(initial01, loss_fn, max_evals=args.max_evals, seed=args.seed)

            after_chain = dict(chain)
            after_chain.update(dict(zip(CONTINUOUS_NAMES, denormalize_continuous(optimized01.tolist()))))
            after_audio = chain_render(source_audio, sr, after_chain)
            after_mae = continuous_mae(optimized01, target, None)
            after_sim = multi_resolution_stft_distance(ref_audio, after_audio)["similarity"]

            per_example.append({
                "id": example_id,
                "n_evals": n_evals,
                "before": {"continuous_mae": before_mae, "similarity": before_sim},
                "after": {"continuous_mae": after_mae, "similarity": after_sim},
            })

            if (idx + 1) % 50 == 0:
                with open(partial_path, "w") as f:
                    json.dump(per_example, f)
                print(f"checkpoint: {idx + 1}/{n_examples} examples done "
                      f"({time.time() - t0:.0f}s elapsed)")

    elapsed = time.time() - t0

    overall = {
        "n_examples": n_examples,
        "max_evals": args.max_evals,
        "seconds": elapsed,
        "continuous_mae_before_mean": float(np.mean([e["before"]["continuous_mae"] for e in per_example])),
        "continuous_mae_after_mean": float(np.mean([e["after"]["continuous_mae"] for e in per_example])),
        "similarity_before_mean": float(np.mean([e["before"]["similarity"] for e in per_example])),
        "similarity_after_mean": float(np.mean([e["after"]["similarity"] for e in per_example])),
        "similarity_improved_fraction": float(np.mean([
            e["after"]["similarity"] > e["before"]["similarity"] for e in per_example
        ])),
    }

    with open(run_dir / "report.json", "w") as f:
        json.dump({"overall": overall, "per_example": per_example}, f, indent=2)
    partial_path.unlink(missing_ok=True)

    print(json.dumps(overall, indent=2))
    print(f"report written to {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
