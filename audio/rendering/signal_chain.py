"""Experiment 2's renderer: overdrive pedal -> amp voicing -> cabinet IR.

Ground truth for "can the model predict a candidate signal chain, not just
knob values on one fixed amp."
"""

import numpy as np
from pedalboard import Pedalboard, Gain, Distortion, Clipping, LowShelfFilter, PeakFilter, HighShelfFilter, Convolution, Limiter

from audio.rendering.amp_models import AMP_NAMES, AMP_SPECS
from audio.rendering.cabinets import CABINET_NAMES, make_ir
from audio.rendering.chain_params import PARAM_MIN, PARAM_MAX

_IR_CACHE = {}
_CONVOLUTION_CACHE = {}


def _lerp(knob, lo, hi):
    t = (knob - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)
    t = min(max(t, 0.0), 1.0)
    return lo + t * (hi - lo)


def _get_ir(cabinet_name, sr):
    key = (cabinet_name, sr)
    if key not in _IR_CACHE:
        _IR_CACHE[key] = make_ir(cabinet_name, sr)
    return _IR_CACHE[key]


def _get_convolution(cabinet_name, sr):
    # Constructing pedalboard.Convolution allocates FFT partition buffers for
    # its impulse response; doing that fresh on every render (hundreds of
    # thousands of times across a CMA-ES sweep) grows memory unbounded. The
    # IR is fixed per cabinet, so one instance per (cabinet, sr) is reused
    # and reset between calls instead.
    key = (cabinet_name, sr)
    if key not in _CONVOLUTION_CACHE:
        ir = _get_ir(cabinet_name, sr)
        _CONVOLUTION_CACHE[key] = Convolution(impulse_response_filename=ir, mix=1.0, sample_rate=sr)
    return _CONVOLUTION_CACHE[key]


def build_board(chain, sr):
    plugins = []

    if chain["overdrive_on"]:
        drive_db = _lerp(chain["overdrive_drive"], 0.0, 24.0)
        level_db = _lerp(chain["overdrive_level"], -6.0, 6.0)
        plugins += [Clipping(threshold_db=-drive_db), Gain(gain_db=level_db)]

    amp_name = AMP_NAMES[chain["amp_id"]] if isinstance(chain["amp_id"], int) else chain["amp_name"]
    amp_spec = AMP_SPECS[amp_name]

    pre_gain_db = _lerp(chain["gain"], -10.0, 30.0)
    bass_db = _lerp(chain["bass"], -12.0, 12.0)
    mid_db = _lerp(chain["mid"], -12.0, 12.0)
    treble_db = _lerp(chain["treble"], -12.0, 12.0)
    presence_db = _lerp(chain["presence"], 0.0, 15.0)
    master_db = _lerp(chain["master"], -24.0, 0.0)

    plugins += [
        Gain(gain_db=pre_gain_db),
        Distortion(drive_db=amp_spec["drive_db"]),
        LowShelfFilter(cutoff_frequency_hz=amp_spec["bass_hz"], gain_db=bass_db, q=0.7),
        PeakFilter(cutoff_frequency_hz=amp_spec["mid_hz"], gain_db=mid_db, q=0.7),
        HighShelfFilter(cutoff_frequency_hz=amp_spec["treble_hz"], gain_db=treble_db, q=0.7),
        HighShelfFilter(cutoff_frequency_hz=amp_spec["presence_hz"], gain_db=presence_db, q=0.7),
        Gain(gain_db=master_db),
    ]

    cabinet_name = CABINET_NAMES[chain["cabinet_id"]] if isinstance(chain["cabinet_id"], int) else chain["cabinet_name"]
    convolution = _get_convolution(cabinet_name, sr)
    convolution.reset()
    plugins.append(convolution)

    plugins.append(Limiter(threshold_db=-0.3, release_ms=50.0))
    return Pedalboard(plugins)


def render(audio, sr, chain):
    board = build_board(chain, sr)
    mono = audio.astype(np.float32)
    processed = board(mono, sr)
    return np.asarray(processed, dtype=np.float32).reshape(-1)
