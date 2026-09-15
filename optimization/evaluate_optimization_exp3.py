"""Does the optimization loop still work on the full Stage 2 rig?

    fixed   pedal on/off states and amp / cabinet / mic held at the CNN's calls
    oracle  those set to the true ones (a ceiling, not a usable method)

Either way CMA-ES refines the knobs, but only the knobs that matter for the
chain being rendered: knobs of pedals that are off change nothing, so
searching them would only spend renders. Seeds are per example (see
evaluate_optimization_exp2.py) and the run resumes from its last checkpoint.

Usage (from repo root):
    python -m optimization.evaluate_optimization_exp3 --config configs/experiment3_300k.yaml --mode fixed --n-examples 1000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from audio.rendering.signal_chain3 import EXP3_SPEC as SPEC, render
from audio.similarity.stft_loss import multi_resolution_stft_distance
from models.tone_predictor.multihead import MultiHeadChainNet
from optimization.optimizer import optimize_generic
from training.exp3_common import chain_from, decode, knob_mae
from training.shards import ShardSet

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--mode", choices=["fixed", "oracle"], default="fixed")
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--max-evals", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = config["dataset"]["sr"]
    test = ShardSet(REPO_ROOT / config["dataset"]["data_root"] / "test", SPEC).load_all(with_audio=True)
    n_examples = min(args.n_examples, len(test["features"]))

    model = MultiHeadChainNet(SPEC).to(device)
    model.load_state_dict(torch.load(REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / args.checkpoint,
                                     map_location=device))
    model.eval()

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"] / "optimization"
    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"report_{args.mode}" + (f"_{args.tag}" if args.tag else "")
    report_path, partial_path = run_dir / f"{name}.json", run_dir / f"{name}_partial.json"

    per_example, prior_seconds = [], 0.0
    if partial_path.exists():
        with open(partial_path) as f:
            partial = json.load(f)
        per_example, prior_seconds = partial["per_example"], partial["seconds"]
        print(f"resuming from {partial_path.name}: {len(per_example)} examples already done", flush=True)

    t0 = time.time() - prior_seconds
    with torch.no_grad():
        for i in range(len(per_example), n_examples):
            pred = model(test["features"][i:i + 1].to(device))
            knobs01, switches, choices = decode(pred, 0)
            true01 = test["continuous"][i].numpy()
            true_switches = test["switches"][i].numpy().astype(np.int8)
            true_choices = test["choices"][i].numpy().astype(np.int8)
            source, ref = test["source"][i], test["audio"][i]

            def similarity(k01, sw, ch):
                return multi_resolution_stft_distance(ref, render(source, sr, chain_from(k01, sw, ch)))

            before = similarity(knobs01, switches, choices)
            if args.mode == "oracle":
                switches, choices = true_switches, true_choices
            active = np.flatnonzero(SPEC.knob_mask(switches))

            def loss_fn(v_active, base=knobs01, sw=switches, ch=choices):
                k01 = base.copy()
                k01[active] = v_active
                d = similarity(k01, sw, ch)
                return d["spectral_convergence"] + d["log_mag_error"]

            example_seed = args.seed * 100_000 + i
            best_active, n_evals = optimize_generic(knobs01[active], loss_fn, max_evals=args.max_evals, seed=example_seed)
            optimized01 = knobs01.copy()
            optimized01[active] = best_active
            after = similarity(optimized01, switches, choices)

            per_example.append({
                "index": i, "seed": example_seed, "n_evals": int(n_evals), "n_active_knobs": int(len(active)),
                "before": {"similarity": before["similarity"], "knob_mae": knob_mae(knobs01, true01, true_switches)},
                "after": {"similarity": after["similarity"], "knob_mae": knob_mae(optimized01, true01, true_switches)},
                "switches_correct": int((switches == true_switches).all()),
                "choices_correct": {c: int(choices[k] == true_choices[k]) for k, c in enumerate(SPEC.choice_names)},
            })
            if (i + 1) % 50 == 0:
                with open(partial_path, "w") as f:
                    json.dump({"per_example": per_example, "seconds": time.time() - t0}, f)
                print(f"checkpoint: {i + 1}/{n_examples} examples done ({time.time() - t0:.0f}s elapsed)", flush=True)

    def mean(fn):
        return float(np.mean([fn(e) for e in per_example]))

    overall = {
        "mode": args.mode, "n_examples": n_examples, "max_evals": args.max_evals, "seconds": time.time() - t0,
        "n_evals_mean": mean(lambda e: e["n_evals"]),
        "similarity_before_mean": mean(lambda e: e["before"]["similarity"]),
        "similarity_after_mean": mean(lambda e: e["after"]["similarity"]),
        "knob_mae_before_mean": mean(lambda e: e["before"]["knob_mae"]),
        "knob_mae_after_mean": mean(lambda e: e["after"]["knob_mae"]),
        "similarity_improved_fraction": mean(lambda e: e["after"]["similarity"] > e["before"]["similarity"]),
        "all_switches_correct": mean(lambda e: e["switches_correct"]),
    }
    with open(report_path, "w") as f:
        json.dump({"overall": overall, "per_example": per_example}, f, indent=2)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
