"""Shared knob definitions for Experiment 4's source/performance variables.

Same shape as audio/rendering/params.py: all knobs on a 0-10 scale.
"""

PARAM_NAMES = ["pickup_position", "dynamics", "tuning_semitones"]
PARAM_MIN = 0.0
PARAM_MAX = 10.0


def random_params(rng):
    return {name: float(rng.uniform(PARAM_MIN, PARAM_MAX)) for name in PARAM_NAMES}


def denormalize(vector):
    return [v * (PARAM_MAX - PARAM_MIN) + PARAM_MIN for v in vector]
