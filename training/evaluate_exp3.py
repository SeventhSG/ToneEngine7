"""Experiment 3 test-set report: can the network read the whole Stage 2 rig
(which pedals are on, which amp, cabinet and mic, and every knob) from
unseen rendered audio?

Usage (from repo root):
    python -m training.evaluate_exp3 --config configs/experiment3_300k.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from audio.rendering.signal_chain3 import EXP3_SPEC as SPEC, render
from audio.similarity.stft_loss import multi_resolution_stft_distance
from models.tone_predictor.multihead import MultiHeadChainNet
from training.exp3_common import chain_from, decode, knob_mae, per_knob_errors
from training.shards import ShardSet

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = config["dataset"]["sr"]
    test = ShardSet(REPO_ROOT / config["dataset"]["data_root"] / "test", SPEC).load_all(with_audio=True)
    n = len(test["features"])

    model = MultiHeadChainNet(SPEC).to(device)
    model.load_state_dict(torch.load(REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / args.checkpoint,
                                     map_location=device))
    model.eval()

    per_example, knob_errors = [], []
    with torch.no_grad():
        for start in range(0, n, 250):
            pred = model(test["features"][start:start + 250].to(device))
            for j in range(len(pred["continuous"])):
                i = start + j
                knobs01, switches, choices = decode(pred, j)
                true_switches = test["switches"][i].numpy().astype(np.int8)
                true_choices = test["choices"][i].numpy()
                true01 = test["continuous"][i].numpy()
                gen = render(test["source"][i], sr, chain_from(knobs01, switches, choices))
                sim = multi_resolution_stft_distance(test["audio"][i], gen)
                per_example.append({
                    "index": i,
                    "knob_mae": knob_mae(knobs01, true01, true_switches),
                    "switch_correct": {s: int(switches[k] == true_switches[k]) for k, s in enumerate(SPEC.switch_names)},
                    "choice_correct": {c: int(choices[k] == true_choices[k]) for k, c in enumerate(SPEC.choice_names)},
                    "audio_similarity": sim["similarity"],
                    "noise_db": float(test["noise_db"][i]),
                })
                knob_errors.append(per_knob_errors(knobs01, true01, true_switches))

    knob_errors = np.array(knob_errors)
    overall = {
        "n_test_examples": n,
        "knob_mae_mean": float(np.mean([e["knob_mae"] for e in per_example])),
        "audio_similarity_mean": float(np.mean([e["audio_similarity"] for e in per_example])),
        "switch_accuracy": {s: float(np.mean([e["switch_correct"][s] for e in per_example])) for s in SPEC.switch_names},
        "choice_accuracy": {c: float(np.mean([e["choice_correct"][c] for e in per_example])) for c in SPEC.choice_names},
        "knob_mae_by_knob": {k: float(np.nanmean(knob_errors[:, j])) for j, k in enumerate(SPEC.continuous)},
    }
    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"] / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "test_report.json", "w") as f:
        json.dump({"overall": overall, "per_example": per_example}, f, indent=2)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
