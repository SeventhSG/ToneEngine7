"""Training batches from a shard pool written by generate_shards_exp2.py.

The pool lives on a spinning disk, where random access to single examples
is slow, so shuffling is done in blocks: each epoch the shards are put in a
random order, `shards_per_block` of them are read sequentially into memory,
and batches are drawn from that block in random order. Small pools (a few
shards) fit in one block and are shuffled exactly.

Targets match training/dataset_exp2.py's ChainDataset, so the same loss and
model code works unchanged.
"""

import json
from pathlib import Path

import numpy as np
import torch

from audio.rendering.chain_params import PARAM_MAX, PARAM_MIN


class ShardPool:
    def __init__(self, pool_dir, n_train, shards_per_block=2):
        self.pool_dir = Path(pool_dir)
        with open(self.pool_dir / "manifest.json") as f:
            self.manifest = json.load(f)

        self.shards = []  # (shard_dir, n_used)
        remaining = n_train
        for shard_dir in sorted(self.pool_dir.glob("shard_*")):
            if remaining <= 0:
                break
            if not (shard_dir / "done.json").exists():
                raise RuntimeError(f"{shard_dir} is incomplete; rerun generate_shards_exp2.py")
            n = int(json.loads((shard_dir / "done.json").read_text())["n"])
            self.shards.append((shard_dir, min(n, remaining)))
            remaining -= min(n, remaining)
        if remaining > 0:
            raise ValueError(f"pool has fewer than {n_train} examples")
        self.n_train = n_train
        self.shards_per_block = shards_per_block
        self._cache = None
        if len(self.shards) <= shards_per_block:
            self._cache = self._load(self.shards)  # whole subset fits in one block: keep it resident

    def __len__(self):
        return self.n_train

    def _load(self, shards):
        # Fill one preallocated array instead of copying each shard and then
        # concatenating, which briefly held every block twice. This machine is
        # tight on commit charge, so the peak matters.
        first = np.load(shards[0][0] / "features.npy", mmap_mode="r")
        features = np.empty((sum(n for _, n in shards),) + first.shape[1:], dtype=first.dtype)
        cont, od, amp, cab = [], [], [], []
        offset = 0
        for shard_dir, n in shards:
            features[offset:offset + n] = np.load(shard_dir / "features.npy", mmap_mode="r")[:n]
            offset += n
            labels = np.load(shard_dir / "labels.npz")
            cont.append(labels["continuous"][:n])
            od.append(labels["overdrive_on"][:n])
            amp.append(labels["amp_id"][:n])
            cab.append(labels["cabinet_id"][:n])
        return {
            "features": torch.from_numpy(features),
            "continuous": torch.from_numpy((np.concatenate(cont) - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)).float(),
            "overdrive_on": torch.from_numpy(np.concatenate(od)).float(),
            "amp_id": torch.from_numpy(np.concatenate(amp)).long(),
            "cabinet_id": torch.from_numpy(np.concatenate(cab)).long(),
        }

    def batches(self, batch_size, epoch, seed=0):
        """Yields (features, target) for one epoch."""
        rng = np.random.default_rng([seed, epoch])
        if self._cache is not None:
            groups = [None]
        else:
            order = rng.permutation(len(self.shards))
            groups = [order[i:i + self.shards_per_block] for i in range(0, len(order), self.shards_per_block)]

        block = None
        for g in groups:
            block = None  # release the previous block before reading the next, so only one is resident
            block = self._cache if g is None else self._load([self.shards[j] for j in g])
            perm = torch.from_numpy(rng.permutation(len(block["features"])))
            for start in range(0, len(perm), batch_size):
                idx = perm[start:start + batch_size]
                target = {k: block[k][idx] for k in ("continuous", "overdrive_on", "amp_id", "cabinet_id")}
                yield block["features"][idx], target
