"""How much of each pedal's on/off accuracy is limited by physics, not the model?

A pedal can be switched on and still change nothing: a noise gate whose
threshold sits below the noise floor never closes, an EQ with every band
near the middle is flat, a low-drive overdrive is almost a volume change.
For each test example where a pedal is truly on, this renders the true
chain again with only that pedal switched off and measures the difference.
The CNN's on/off accuracy is then split by whether the pedal was audible.

Usage (from repo root):
    python -m evaluation.identifiability_exp3 --run exp3_1m [--threshold 0.02]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from audio.rendering.signal_chain3 import EXP3_SPEC as SPEC, render
from audio.similarity.stft_loss import multi_resolution_stft_distance
from training.exp3_common import chain_from
from training.shards import ShardSet

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/generated/exp3")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="1 - similarity above which switching the pedal off counts as audible")
    args = parser.parse_args()

    run_dir = REPO_ROOT / "training" / "experiments" / args.run / "eval"
    with open(run_dir / "test_report.json") as f:
        per_example = json.load(f)["per_example"]
    test = ShardSet(REPO_ROOT / args.data_root / "test", SPEC).load_all(with_audio=True)

    results = {}
    for s_idx, s in enumerate(SPEC.switch_names):
        audible, correct_on, correct_off = [], [], []
        for e in per_example:
            i = e["index"]
            true_switches = test["switches"][i].numpy().astype(np.int8)
            if not true_switches[s_idx]:
                correct_off.append(e["switch_correct"][s])
                continue
            chain = chain_from(test["continuous"][i].numpy(), true_switches, test["choices"][i].numpy())
            flipped = dict(chain, **{s: False})
            d = 1 - multi_resolution_stft_distance(render(test["source"][i], 16000, chain),
                                                   render(test["source"][i], 16000, flipped))["similarity"]
            audible.append(d > args.threshold)
            correct_on.append(e["switch_correct"][s])
        audible, correct_on = np.array(audible), np.array(correct_on)
        results[s] = {
            "n_on": int(len(audible)), "audible_fraction": float(audible.mean()),
            "accuracy_when_on_and_audible": float(correct_on[audible].mean()) if audible.any() else None,
            "accuracy_when_on_but_inaudible": float(correct_on[~audible].mean()) if (~audible).any() else None,
            "accuracy_when_off": float(np.mean(correct_off)),
        }
        r = results[s]
        fmt = lambda v: "n/a" if v is None else f"{v:.1%}"
        print(f"{s:13s} on in {r['n_on']} examples, audible in {r['audible_fraction']:.0%} "
              f"({int((~audible).sum())} inaudible); accuracy: on+audible {fmt(r['accuracy_when_on_and_audible'])}, "
              f"on+inaudible {fmt(r['accuracy_when_on_but_inaudible'])}, off {fmt(r['accuracy_when_off'])}")

    with open(run_dir / "identifiability.json", "w") as f:
        json.dump({"threshold": args.threshold, "switches": results}, f, indent=2)


if __name__ == "__main__":
    main()
