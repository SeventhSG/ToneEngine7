"""Does the optimization loop help on Experiment 2's signal chain?

Three modes, all refining the continuous knobs with CMA-ES under the same
render budget, differing only in how the discrete choices (overdrive on/off,
amp, cabinet) are handled:

    fixed   choices held at the CNN's prediction (argmax), never revisited
    search  several choices are tried and the one reaching the lowest loss
            is kept and refined further
            (optimization.optimizer.optimize_with_choices)
    oracle  choices set to the true ones; not a usable method, only a ceiling
            showing how much a perfect discrete search could gain

The search defaults (every amp, 300 renders each, 1800 in total) were picked
on the validation split (--split val); see the README for the sweep. The
search needs a larger budget than the fixed mode's default of 300 to pay off,
so compare modes at the same --max-evals.

Usage (from repo root):
    python -m optimization.evaluate_optimization_exp2 --config configs/experiment2.yaml --mode fixed --n-examples 1000
    python -m optimization.evaluate_optimization_exp2 --config configs/experiment2.yaml --mode search --max-evals 1800 --n-examples 200
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml

from audio.features.spectrogram import LogMelFeature
from audio.preprocessing.io import load_wav
from audio.rendering.chain_params import CONTINUOUS_NAMES, N_AMPS, N_CABINETS, denormalize_continuous
from audio.rendering.signal_chain import render as chain_render
from audio.similarity.stft_loss import multi_resolution_stft_distance
from models.tone_predictor.chain_model import ChainPredictor
from optimization.optimizer import optimize_generic, optimize_with_choices
from training.dataset_exp2 import ChainDataset
from training.evaluate_exp2 import continuous_mae
from training.train_exp2 import pred_to_chain

REPO_ROOT = Path(__file__).resolve().parents[1]

REPORT_NAMES = {"fixed": "report.json", "search": "report_search.json", "oracle": "report_oracle.json"}


def ranked_choices(pred, idx, vary="all"):
    """(overdrive_on, amp_id, cabinet_id) combinations, most likely first
    under the CNN's heads, treated as independent.

    vary="all" ranks every combination. vary="amp" tries every amp but keeps
    the CNN's overdrive and cabinet calls. The cabinet call is already ~100%
    right. Overdrive on/off is only weakly identifiable from the rendered
    audio: given the true amp and cabinet and 300 renders each, the true
    overdrive state reached the lower loss in 62 of 100 validation examples
    (a low-drive pedal is mostly a level change the amp's gain knob can
    reproduce), so searching over it mostly spends budget.
    """
    amp_logp = F.log_softmax(pred["amp_logits"][idx], dim=-1).cpu().numpy()
    cabinet_logp = F.log_softmax(pred["cabinet_logits"][idx], dim=-1).cpu().numpy()
    od_logit = pred["overdrive_logit"][idx]
    od_logp = {True: F.logsigmoid(od_logit).item(), False: F.logsigmoid(-od_logit).item()}

    if vary == "amp":
        od_options = [od_logit.item() > 0]
        cabinet_options = [int(cabinet_logp.argmax())]
    else:
        od_options, cabinet_options = [True, False], range(N_CABINETS)

    combos = itertools.product(od_options, range(N_AMPS), cabinet_options)
    scored = [((od, amp, cab), od_logp[od] + amp_logp[amp] + cabinet_logp[cab]) for od, amp, cab in combos]
    return [choice for choice, _ in sorted(scored, key=lambda cs: -cs[1])]


def chain_from(choice, vector01):
    overdrive_on, amp_id, cabinet_id = choice
    chain = dict(zip(CONTINUOUS_NAMES, denormalize_continuous(list(vector01))))
    chain.update({"overdrive_on": bool(overdrive_on), "amp_id": int(amp_id), "cabinet_id": int(cabinet_id)})
    return chain


def choice_correct(choice, target):
    overdrive_on, amp_id, cabinet_id = choice
    return {
        "overdrive": int(bool(overdrive_on) == bool(target["overdrive_on"].item() > 0.5)),
        "amp": int(amp_id == target["amp_id"].item()),
        "cabinet": int(cabinet_id == target["cabinet_id"].item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--mode", choices=list(REPORT_NAMES), default="fixed")
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--max-evals", type=int, default=300)
    parser.add_argument("--n-candidates", type=int, default=3,
                        help="search mode: how many of the CNN's top choice combinations to try")
    parser.add_argument("--vary", choices=["all", "amp"], default="amp",
                        help="search mode: which discrete choices the search may change")
    parser.add_argument("--screen-evals", type=int, default=300, help="search mode: renders per candidate in round 1")
    parser.add_argument("--keep", type=int, default=1, help="search mode: candidates surviving round 1")
    parser.add_argument("--mid-evals", type=int, default=0, help="search mode: renders per survivor in round 2")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="tune the search schedule on val, report on test")
    parser.add_argument("--tag", type=str, default="", help="appended to the report file name")
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
    test_ds = ChainDataset(dataset_dir, args.split, feature_fn, sr)
    n_examples = min(args.n_examples, len(test_ds)) if args.n_examples > 0 else len(test_ds)

    checkpoint_path = REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / args.checkpoint
    model = ChainPredictor().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"] / "optimization"
    if args.split != "test":
        run_dir = run_dir / f"{args.split}_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_name = REPORT_NAMES[args.mode]
    if args.tag:
        report_name = report_name.replace(".json", f"_{args.tag}.json")
    report_path = run_dir / report_name
    partial_path = report_path.with_name(report_path.stem + "_partial.json")

    # Resume from the last checkpoint if a previous run of this exact report
    # was killed partway (this machine runs short on RAM from other apps).
    per_example, prior_seconds = [], 0.0
    if partial_path.exists():
        with open(partial_path) as f:
            partial = json.load(f)
        per_example, prior_seconds = partial["per_example"], partial["seconds"]
        print(f"resuming from {partial_path.name}: {len(per_example)} examples already done", flush=True)

    process = psutil.Process()
    t0 = time.time() - prior_seconds
    with torch.no_grad():
        for idx in range(len(per_example), n_examples):
            feature, target, example_id = test_ds[idx]
            pred = model(feature.unsqueeze(0).to(device))

            example_dir = test_ds.example_dir(idx)
            source_audio, _ = load_wav(example_dir / "input.wav", sr=sr)
            ref_audio, _ = load_wav(example_dir / "audio.wav", sr=sr)

            def loss_fn(choice, vector01):
                gen_audio = chain_render(source_audio, sr, chain_from(choice, vector01))
                result = multi_resolution_stft_distance(ref_audio, gen_audio)
                return result["spectral_convergence"] + result["log_mag_error"]

            cnn_chain = pred_to_chain(pred, 0)
            cnn_choice = (cnn_chain["overdrive_on"], cnn_chain["amp_id"], cnn_chain["cabinet_id"])
            initial01 = pred["continuous"][0].cpu().numpy()

            before_audio = chain_render(source_audio, sr, cnn_chain)
            before_mae = continuous_mae(initial01, target, None)
            before_sim = multi_resolution_stft_distance(ref_audio, before_audio)["similarity"]

            # One seed per example, derived from the base seed: still reproducible, but the
            # average over examples also averages over many CMA-ES seeds. A single shared seed
            # gives every example the same random draws, so the whole run shifts together if
            # that one sequence happens to be lucky or unlucky. The same example gets the same
            # seed in every mode, which keeps fixed / search / oracle comparisons paired.
            example_seed = args.seed * 100_000 + idx
            record = {"id": example_id, "seed": example_seed}
            if args.mode == "search":
                candidates = [(c, initial01) for c in ranked_choices(pred, 0, args.vary)[:args.n_candidates]]
                after_choice, optimized01, n_evals, trace = optimize_with_choices(
                    candidates, loss_fn, max_evals=args.max_evals, keep=args.keep,
                    screen_evals=args.screen_evals, mid_evals=args.mid_evals, seed=example_seed,
                )
                record["screen_rank_of_winner"] = [c for c, _ in candidates].index(after_choice)
                record["trace"] = {k: [[list(map(int, c)), float(l)] for c, l in v] for k, v in trace.items()}
            else:
                if args.mode == "oracle":
                    after_choice = (bool(target["overdrive_on"].item() > 0.5),
                                    int(target["amp_id"].item()), int(target["cabinet_id"].item()))
                else:
                    after_choice = cnn_choice
                optimized01, n_evals = optimize_generic(
                    initial01, lambda v, c=after_choice: loss_fn(c, v),
                    max_evals=args.max_evals, seed=example_seed,
                )

            after_audio = chain_render(source_audio, sr, chain_from(after_choice, optimized01))
            after_mae = continuous_mae(optimized01, target, None)
            after_sim = multi_resolution_stft_distance(ref_audio, after_audio)["similarity"]

            record.update({
                "n_evals": int(n_evals),
                "before": {"continuous_mae": before_mae, "similarity": before_sim,
                           "choice": list(map(int, cnn_choice)), "correct": choice_correct(cnn_choice, target)},
                "after": {"continuous_mae": after_mae, "similarity": after_sim,
                          "choice": list(map(int, after_choice)), "correct": choice_correct(after_choice, target)},
            })
            per_example.append(record)

            if (idx + 1) % 50 == 0:
                with open(partial_path, "w") as f:
                    json.dump({"per_example": per_example, "seconds": time.time() - t0}, f)
                print(f"checkpoint: {idx + 1}/{n_examples} examples done "
                      f"({time.time() - t0:.0f}s elapsed, {process.memory_info().rss / 2**20:.0f} MB resident)",
                      flush=True)

    elapsed = time.time() - t0

    def mean(fn):
        return float(np.mean([fn(e) for e in per_example]))

    overall = {
        "mode": args.mode,
        "split": args.split,
        "n_examples": n_examples,
        "max_evals": args.max_evals,
        "n_evals_mean": mean(lambda e: e["n_evals"]),
        "seconds": elapsed,
        "continuous_mae_before_mean": mean(lambda e: e["before"]["continuous_mae"]),
        "continuous_mae_after_mean": mean(lambda e: e["after"]["continuous_mae"]),
        "similarity_before_mean": mean(lambda e: e["before"]["similarity"]),
        "similarity_after_mean": mean(lambda e: e["after"]["similarity"]),
        "similarity_improved_fraction": mean(lambda e: e["after"]["similarity"] > e["before"]["similarity"]),
    }
    for stage in ("before", "after"):
        for name in ("overdrive", "amp", "cabinet"):
            overall[f"{name}_accuracy_{stage}"] = mean(lambda e: e[stage]["correct"][name])
    if args.mode == "search":
        overall["schedule"] = {"vary": args.vary, "n_candidates": args.n_candidates, "screen_evals": args.screen_evals,
                               "keep": args.keep, "mid_evals": args.mid_evals}
        overall["winner_changed_choice_fraction"] = mean(lambda e: e["screen_rank_of_winner"] != 0)

    with open(report_path, "w") as f:
        json.dump({"overall": overall, "per_example": per_example}, f, indent=2)
    partial_path.unlink(missing_ok=True)

    print(json.dumps(overall, indent=2))
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
