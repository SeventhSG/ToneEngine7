"""Parameter spec for Experiment 2's candidate signal chain:

    overdrive pedal (on/off) -> amp (one of 3 voicings, 6 knobs) -> cabinet IR (one of 4)

This is a mixed continuous / categorical / binary target, unlike
Experiment 1's pure continuous knobs.
"""

from audio.rendering.amp_models import AMP_NAMES
from audio.rendering.cabinets import CABINET_NAMES

CONTINUOUS_NAMES = ["overdrive_drive", "overdrive_level", "gain", "bass", "mid", "treble", "presence", "master"]
PARAM_MIN = 0.0
PARAM_MAX = 10.0

N_AMPS = len(AMP_NAMES)
N_CABINETS = len(CABINET_NAMES)


def random_chain(rng):
    overdrive_on = bool(rng.uniform() < 0.5)
    overdrive_drive = float(rng.uniform(PARAM_MIN, PARAM_MAX)) if overdrive_on else 0.0
    overdrive_level = float(rng.uniform(PARAM_MIN, PARAM_MAX)) if overdrive_on else 0.0

    continuous = {
        "overdrive_drive": overdrive_drive,
        "overdrive_level": overdrive_level,
        "gain": float(rng.uniform(PARAM_MIN, PARAM_MAX)),
        "bass": float(rng.uniform(PARAM_MIN, PARAM_MAX)),
        "mid": float(rng.uniform(PARAM_MIN, PARAM_MAX)),
        "treble": float(rng.uniform(PARAM_MIN, PARAM_MAX)),
        "presence": float(rng.uniform(PARAM_MIN, PARAM_MAX)),
        "master": float(rng.uniform(PARAM_MIN, PARAM_MAX)),
    }
    amp_id = int(rng.integers(0, N_AMPS))
    cabinet_id = int(rng.integers(0, N_CABINETS))

    return {
        "overdrive_on": overdrive_on,
        "amp_id": amp_id,
        "amp_name": AMP_NAMES[amp_id],
        "cabinet_id": cabinet_id,
        "cabinet_name": CABINET_NAMES[cabinet_id],
        **continuous,
    }


def continuous_vector(chain):
    return [chain[name] for name in CONTINUOUS_NAMES]


def normalize_continuous(vector):
    return [(v - PARAM_MIN) / (PARAM_MAX - PARAM_MIN) for v in vector]


def denormalize_continuous(vector):
    return [v * (PARAM_MAX - PARAM_MIN) + PARAM_MIN for v in vector]
