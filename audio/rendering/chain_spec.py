"""Declarative description of a candidate signal chain.

A chain has three kinds of parameters:

    continuous  knobs on a 0-10 scale
    switches    pedals that are on or off; each gates a list of knobs, which
                are meaningless (and masked out of losses and metrics) when
                the pedal is off
    choices     categorical picks, e.g. which amp, cabinet, or microphone

Experiment 2 predates this and keeps its own hand-written chain_params.py so
its published results stay reproducible; Experiment 3 onward is built on a
ChainSpec, so adding gear means extending a spec rather than rewriting the
dataset, model heads, loss, and metrics.
"""

import numpy as np

PARAM_MIN = 0.0
PARAM_MAX = 10.0


class ChainSpec:
    def __init__(self, continuous, switches, choices):
        self.continuous = list(continuous)
        self.switches = dict(switches)
        self.choices = dict(choices)
        gated = [k for knobs in self.switches.values() for k in knobs]
        unknown = [k for k in gated if k not in self.continuous]
        if unknown:
            raise ValueError(f"switches gate unknown knobs: {unknown}")
        self.switch_names = list(self.switches)
        self.choice_names = list(self.choices)
        self._gate_of = {k: s for s, knobs in self.switches.items() for k in knobs}

    def random_chain(self, rng):
        chain = {s: bool(rng.uniform() < 0.5) for s in self.switch_names}
        for k in self.continuous:
            chain[k] = float(rng.uniform(PARAM_MIN, PARAM_MAX))
        for s, knobs in self.switches.items():
            if not chain[s]:
                for k in knobs:
                    chain[k] = 0.0
        for c, options in self.choices.items():
            chain[c] = int(rng.integers(0, len(options)))
        return chain

    def knob_mask(self, switch_values):
        """1 for knobs that matter given these switch states, 0 for knobs of
        pedals that are off. switch_values: sequence aligned with switch_names."""
        on = dict(zip(self.switch_names, switch_values))
        return np.array([0.0 if k in self._gate_of and not on[self._gate_of[k]] else 1.0
                         for k in self.continuous], dtype=np.float32)

    def to_arrays(self, chain):
        continuous = np.array([chain[k] for k in self.continuous], dtype=np.float32)
        switches = np.array([int(chain[s]) for s in self.switch_names], dtype=np.int8)
        choices = np.array([chain[c] for c in self.choice_names], dtype=np.int8)
        return continuous, switches, choices

    def from_arrays(self, continuous, switches, choices):
        chain = {k: float(v) for k, v in zip(self.continuous, continuous)}
        chain.update({s: bool(v) for s, v in zip(self.switch_names, switches)})
        chain.update({c: int(v) for c, v in zip(self.choice_names, choices)})
        return chain


def normalize(values):
    return (np.asarray(values, dtype=np.float32) - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)


def denormalize(values01):
    return np.asarray(values01, dtype=np.float32) * (PARAM_MAX - PARAM_MIN) + PARAM_MIN
