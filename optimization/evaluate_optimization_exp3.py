"""Does the optimization loop still work on the full Stage 2 rig?

    fixed   pedal on/off states and amp / cabinet / mic held at the CNN's calls
    search  a few of the CNN's top switch/choice combinations are tried and the
            one reaching the lowest loss is kept and refined further
            (optimization.optimizer.optimize_with_choices)
    oracle  those set to the true ones (a ceiling, not a usable method)

Either way CMA-ES refines the knobs, but only the knobs that matter for the
chain being rendered: knobs of pedals that are off change nothing, so
searching them would only spend renders. In search mode this mask can differ
per candidate, since a candidate may flip a switch the CNN had off. Seeds are
per example (see evaluate_optimization_exp2.py) and the run resumes from its
last checkpoint.

Usage (from repo root):
    python -m optimization.evaluate_optimization_exp3 --config configs/experiment3_300k.yaml --mode fixed --n-examples 1000
    python -m optimization.evaluate_optimization_exp3 --config configs/experiment3_1m.yaml --mode search --vary amp --max-evals 1800 --n-examples 200
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from audio.rendering.signal_chain3 import EXP3_SPEC as SPEC, render
from audio.similarity.stft_loss import multi_resolution_stft_distance
from models.tone_predictor.multihead import MultiHeadChainNet
from optimization.optimizer import optimize_generic, optimize_with_choices
from training.exp3_common import chain_from, decode, knob_mae
from training.shards import ShardSet

REPO_ROOT = Path(__file__).resolve().parents[1]


def ranked_choices(pred, idx, vary):
    """Combinations of the varied switch/choice axes, most likely first under
    the CNN's own heads (treated as independent), with every other axis held
    at the CNN's decode.

    vary: list of names from SPEC.switch_names / SPEC.choice_names, e.g.
    ["amp"] or ["amp", "comp_on"]. Mirrors evaluate_optimization_exp2's
    ranked_choices, generalized from a hardcoded tuple to a ChainSpec.
    """
    base_switches = (pred["switch_logits"][idx] > 0).cpu().numpy().astype(np.int8)
    base_choices = np.array([int(logits[idx].argmax()) for logits in pred["choice_logits"]], dtype=np.int8)

    axis_options, axis_logp = {}, {}
    for s in vary:
        if s in SPEC.switch_names:
            j = SPEC.switch_names.index(s)
            logit = pred["switch_logits"][idx, j]
            axis_options[s] = [1, 0]
            axis_logp[s] = {1: F.logsigmoid(logit).item(), 0: F.logsigmoid(-logit).item()}
        else:
            j = SPEC.choice_names.index(s)
            logp = F.log_softmax(pred["choice_logits"][j][idx], dim=-1).cpu().numpy()
            axis_options[s] = list(range(len(logp)))
            axis_logp[s] = {v: float(logp[v]) for v in axis_options[s]}

    combos = itertools.product(*(axis_options[s] for s in vary))
    scored = []
    for values in combos:
        switches, choices = base_switches.copy(), base_choices.copy()
        for s, v in zip(vary, values):
            if s in SPEC.switch_names:
                switches[SPEC.switch_names.index(s)] = v
            else:
                choices[SPEC.choice_names.index(s)] = v
        logp = sum(axis_logp[s][v] for s, v in zip(vary, values))
        scored.append(((switches, choices), logp))
    return [choice for choice, _ in sorted(scored, key=lambda cs: -cs[1])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--mode", choices=["fixed", "search", "oracle"], default="fixed")
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--max-evals", type=int, default=300)
    parser.add_argument("--vary", type=str, default="amp",
                        help="search mode: comma-separated switch/choice names to search over")
    parser.add_argument("--n-candidates", type=int, default=3,
                        help="search mode: how many of the CNN's top combinations to try")
    parser.add_argument("--screen-evals", type=int, default=300, help="search mode: renders per candidate in round 1")
    parser.add_argument("--keep", type=int, default=1, help="search mode: candidates surviving round 1")
    parser.add_argument("--mid-evals", type=int, default=0, help="search mode: renders per survivor in round 2")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="tune the search schedule on val, report on test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = config["dataset"]["sr"]
    test = ShardSet(REPO_ROOT / config["dataset"]["data_root"] / args.split, SPEC).load_all(with_audio=True)
    n_examples = min(args.n_examples, len(test["features"]))

    model = MultiHeadChainNet(SPEC).to(device)
    model.load_state_dict(torch.load(REPO_ROOT / "models" / "checkpoints" / config["experiment_name"] / args.checkpoint,
                                     map_location=device))
    model.eval()

    run_dir = REPO_ROOT / "training" / "experiments" / config["experiment_name"] / "optimization"
    if args.split != "test":
        run_dir = run_dir / f"{args.split}_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"report_{args.mode}" + (f"_{args.tag}" if args.tag else "")
    report_path, partial_path = run_dir / f"{name}.json", run_dir / f"{name}_partial.json"

    per_example, prior_seconds = [], 0.0
    if partial_path.exists():
        with open(partial_path) as f:
            partial = json.load(f)
        per_example, prior_seconds = partial["per_example"], partial["seconds"]
        print(f"resuming from {partial_path.name}: {len(per_example)} examples already done", flush=True)

    vary_axes = args.vary.split(",")

    t0 = time.time() - prior_seconds
    with torch.no_grad():
        for i in range(len(per_example), n_examples):
            pred = model(test["features"][i:i + 1].to(device))
            knobs01, cnn_switches, cnn_choices = decode(pred, 0)
            true01 = test["continuous"][i].numpy()
            true_switches = test["switches"][i].numpy().astype(np.int8)
            true_choices = test["choices"][i].numpy().astype(np.int8)
            source, ref = test["source"][i], test["audio"][i]

            def similarity(k01, sw, ch):
                return multi_resolution_stft_distance(ref, render(source, sr, chain_from(k01, sw, ch)))

            before = similarity(knobs01, cnn_switches, cnn_choices)
            example_seed = args.seed * 100_000 + i
            screen_rank_of_winner, trace = None, None

            if args.mode == "search":
                cnn_active = set(np.flatnonzero(SPEC.knob_mask(cnn_switches)).tolist())

                def loss_fn(choice, v_active, base=knobs01):
                    cand_sw = np.array(choice[0], dtype=np.int8)
                    cand_ch = np.array(choice[1], dtype=np.int8)
                    idx = np.flatnonzero(SPEC.knob_mask(cand_sw))
                    k01 = base.copy()
                    k01[idx] = v_active
                    d = similarity(k01, cand_sw, cand_ch)
                    return d["spectral_convergence"] + d["log_mag_error"]

                candidates = []
                for cand_sw, cand_ch in ranked_choices(pred, 0, vary_axes)[:args.n_candidates]:
                    idx = np.flatnonzero(SPEC.knob_mask(cand_sw))
                    init = np.array([knobs01[k] if k in cnn_active else 0.5 for k in idx], dtype=np.float32)
                    candidates.append(((tuple(int(x) for x in cand_sw), tuple(int(x) for x in cand_ch)), init))

                winner, best_active, n_evals, raw_trace = optimize_with_choices(
                    candidates, loss_fn, max_evals=args.max_evals, keep=args.keep,
                    screen_evals=args.screen_evals, mid_evals=args.mid_evals, seed=example_seed,
                )
                screen_rank_of_winner = [c for c, _ in candidates].index(winner)
                trace = {k: [[[list(c[0]), list(c[1])], float(l)] for c, l in v] for k, v in raw_trace.items()}
                switches, choices = np.array(winner[0], dtype=np.int8), np.array(winner[1], dtype=np.int8)
                active = np.flatnonzero(SPEC.knob_mask(switches))
                optimized01 = knobs01.copy()
                optimized01[active] = best_active
            else:
                switches, choices = (true_switches, true_choices) if args.mode == "oracle" else (cnn_switches, cnn_choices)
                active = np.flatnonzero(SPEC.knob_mask(switches))

                def loss_fn(v_active, base=knobs01, sw=switches, ch=choices, act=active):
                    k01 = base.copy()
                    k01[act] = v_active
                    d = similarity(k01, sw, ch)
                    return d["spectral_convergence"] + d["log_mag_error"]

                best_active, n_evals = optimize_generic(knobs01[active], loss_fn, max_evals=args.max_evals, seed=example_seed)
                optimized01 = knobs01.copy()
                optimized01[active] = best_active

            after = similarity(optimized01, switches, choices)

            record = {
                "index": i, "seed": example_seed, "n_evals": int(n_evals), "n_active_knobs": int(len(active)),
                "before": {"similarity": before["similarity"], "knob_mae": knob_mae(knobs01, true01, true_switches)},
                "after": {"similarity": after["similarity"], "knob_mae": knob_mae(optimized01, true01, true_switches)},
                "switches_correct_before": int((cnn_switches == true_switches).all()),
                "choices_correct_before": {c: int(cnn_choices[k] == true_choices[k]) for k, c in enumerate(SPEC.choice_names)},
                "switches_correct": int((switches == true_switches).all()),
                "choices_correct": {c: int(choices[k] == true_choices[k]) for k, c in enumerate(SPEC.choice_names)},
            }
            if screen_rank_of_winner is not None:
                record["screen_rank_of_winner"] = screen_rank_of_winner
                record["trace"] = trace
            per_example.append(record)
            if (i + 1) % 50 == 0:
                with open(partial_path, "w") as f:
                    json.dump({"per_example": per_example, "seconds": time.time() - t0}, f)
                print(f"checkpoint: {i + 1}/{n_examples} examples done ({time.time() - t0:.0f}s elapsed)", flush=True)

    def mean(fn):
        return float(np.mean([fn(e) for e in per_example]))

    overall = {
        "mode": args.mode, "split": args.split, "n_examples": n_examples, "max_evals": args.max_evals,
        "seconds": time.time() - t0,
        "n_evals_mean": mean(lambda e: e["n_evals"]),
        "similarity_before_mean": mean(lambda e: e["before"]["similarity"]),
        "similarity_after_mean": mean(lambda e: e["after"]["similarity"]),
        "knob_mae_before_mean": mean(lambda e: e["before"]["knob_mae"]),
        "knob_mae_after_mean": mean(lambda e: e["after"]["knob_mae"]),
        "similarity_improved_fraction": mean(lambda e: e["after"]["similarity"] > e["before"]["similarity"]),
        "all_switches_correct_before": mean(lambda e: e["switches_correct_before"]),
        "all_switches_correct": mean(lambda e: e["switches_correct"]),
    }
    if args.mode == "search":
        overall["vary"] = vary_axes
        overall["schedule"] = {"n_candidates": args.n_candidates, "screen_evals": args.screen_evals,
                               "keep": args.keep, "mid_evals": args.mid_evals}
        overall["winner_changed_choice_fraction"] = mean(lambda e: e["screen_rank_of_winner"] != 0)
    with open(report_path, "w") as f:
        json.dump({"overall": overall, "per_example": per_example}, f, indent=2)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
