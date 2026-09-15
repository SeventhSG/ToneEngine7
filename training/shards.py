"""Load spec-driven shard sets written by generate_exp3.py.

Training pools are read in shuffled blocks (see pool_exp2.py for why);
validation and test sets are small and loaded whole, including the source
and reference audio when the shards kept it.
"""

import json
from pathlib import Path

import numpy as np
import torch

from audio.rendering.chain_spec import normalize


def knob_mask(spec, switches):
    """[n, n_knobs] mask: 0 for knobs whose pedal is off, 1 otherwise.
    switches: [n, n_switches] array or tensor of 0/1."""
    switches = torch.as_tensor(switches).float()
    mask = torch.ones(switches.shape[0], len(spec.continuous))
    for s_idx, s in enumerate(spec.switch_names):
        for k in spec.switches[s]:
            mask[:, spec.continuous.index(k)] = switches[:, s_idx]
    return mask


class ShardSet:
    def __init__(self, split_dir, spec, n=None, shards_per_block=2):
        self.split_dir = Path(split_dir)
        self.spec = spec
        with open(self.split_dir / "manifest.json") as f:
            self.manifest = json.load(f)
        if self.manifest["continuous"] != spec.continuous:
            raise ValueError("shard set was generated with a different chain spec")

        self.shards = []
        remaining = n if n is not None else float("inf")
        for shard_dir in sorted(self.split_dir.glob("shard_*")):
            if remaining <= 0:
                break
            done = shard_dir / "done.json"
            if not done.exists():
                if n is None:
                    break
                raise RuntimeError(f"{shard_dir} is incomplete; rerun generate_exp3.py")
            size = int(json.loads(done.read_text())["n"])
            take = int(min(size, remaining))
            self.shards.append((shard_dir, take))
            remaining -= take
        if n is not None and remaining > 0:
            raise ValueError(f"{split_dir} has fewer than {n} finished examples")
        self.n = sum(t for _, t in self.shards)
        self.shards_per_block = shards_per_block
        self._cache = self._load(self.shards) if len(self.shards) <= shards_per_block else None

    def __len__(self):
        return self.n

    def _load(self, shards, with_audio=False):
        first = np.load(shards[0][0] / "features.npy", mmap_mode="r")
        features = np.empty((sum(t for _, t in shards),) + first.shape[1:], dtype=first.dtype)
        labels = {"continuous": [], "switches": [], "choices": [], "noise_db": []}
        audio = {"source": [], "audio": []}
        offset = 0
        for shard_dir, take in shards:
            features[offset:offset + take] = np.load(shard_dir / "features.npy", mmap_mode="r")[:take]
            offset += take
            arrays = np.load(shard_dir / "labels.npz")
            for k in labels:
                labels[k].append(arrays[k][:take])
            if with_audio:
                for k in audio:
                    audio[k].append(np.load(shard_dir / f"{k}.npy", mmap_mode="r")[:take].copy())
        switches = np.concatenate(labels["switches"])
        block = {
            "features": torch.from_numpy(features),
            "continuous": torch.from_numpy(normalize(np.concatenate(labels["continuous"]))),
            "switches": torch.from_numpy(switches).float(),
            "choices": torch.from_numpy(np.concatenate(labels["choices"])).long(),
            "mask": knob_mask(self.spec, switches),
            "noise_db": torch.from_numpy(np.concatenate(labels["noise_db"])),
        }
        if with_audio:
            block.update({k: np.concatenate(v) for k, v in audio.items()})
        return block

    def load_all(self, with_audio=False):
        """Everything in memory at once; meant for the 1,000-example val and test sets."""
        return self._load(self.shards, with_audio=with_audio)

    def batches(self, batch_size, epoch, seed=0):
        rng = np.random.default_rng([seed, epoch])
        if self._cache is not None:
            groups = [None]
        else:
            order = rng.permutation(len(self.shards))
            groups = [order[i:i + self.shards_per_block] for i in range(0, len(order), self.shards_per_block)]
        block = None
        for g in groups:
            block = None  # release the previous block before reading the next one
            block = self._cache if g is None else self._load([self.shards[j] for j in g])
            perm = torch.from_numpy(rng.permutation(len(block["features"])))
            for start in range(0, len(perm), batch_size):
                idx = perm[start:start + batch_size]
                yield block["features"][idx], {k: block[k][idx] for k in ("continuous", "switches", "choices", "mask")}
