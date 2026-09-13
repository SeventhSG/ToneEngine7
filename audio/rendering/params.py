"""Shared knob definitions for the Experiment 1 virtual amp.

All knobs live on a 0-10 scale, matching a physical amp panel and the
example values in INSTRUCTIONS.md (gain: 7.32, bass: 4.12, ...).
"""

PARAM_NAMES = ["gain", "bass", "mid", "treble", "presence", "master"]
PARAM_MIN = 0.0
PARAM_MAX = 10.0


def random_params(rng):
    return {name: float(rng.uniform(PARAM_MIN, PARAM_MAX)) for name in PARAM_NAMES}


def params_to_vector(params):
    return [params[name] for name in PARAM_NAMES]


def vector_to_params(vector):
    return {name: float(v) for name, v in zip(PARAM_NAMES, vector)}


def normalize(vector):
    return [(v - PARAM_MIN) / (PARAM_MAX - PARAM_MIN) for v in vector]


def denormalize(vector):
    return [v * (PARAM_MAX - PARAM_MIN) + PARAM_MIN for v in vector]
