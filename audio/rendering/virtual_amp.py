"""A controllable virtual amp used as the ground truth renderer for Experiment 1.

Signal chain: pre-gain -> fixed-drive distortion -> 3-band tone stack ->
presence shelf -> master output gain -> safety limiter.

Knob values are 0-10. Each knob maps to a distinct DSP stage so the audio
actually carries separable information about every parameter:

- gain:     how hard the signal is driven into the distortion stage
- bass/mid/treble: a shelf/peak/shelf tone stack, 5 = flat
- presence: a high shelf above the tone stack, boost only
- master:   final output level, never boosts past unity
"""

import numpy as np
from pedalboard import Pedalboard, Gain, Distortion, LowShelfFilter, PeakFilter, HighShelfFilter, Limiter

from audio.rendering.params import PARAM_MIN, PARAM_MAX


def _lerp(knob, lo, hi):
    t = (knob - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)
    t = min(max(t, 0.0), 1.0)
    return lo + t * (hi - lo)


def build_board(params):
    pre_gain_db = _lerp(params["gain"], -10.0, 30.0)
    bass_db = _lerp(params["bass"], -12.0, 12.0)
    mid_db = _lerp(params["mid"], -12.0, 12.0)
    treble_db = _lerp(params["treble"], -12.0, 12.0)
    presence_db = _lerp(params["presence"], 0.0, 15.0)
    master_db = _lerp(params["master"], -24.0, 0.0)

    return Pedalboard([
        Gain(gain_db=pre_gain_db),
        Distortion(drive_db=12.0),
        LowShelfFilter(cutoff_frequency_hz=150.0, gain_db=bass_db, q=0.7),
        PeakFilter(cutoff_frequency_hz=800.0, gain_db=mid_db, q=0.7),
        HighShelfFilter(cutoff_frequency_hz=3000.0, gain_db=treble_db, q=0.7),
        HighShelfFilter(cutoff_frequency_hz=5500.0, gain_db=presence_db, q=0.7),
        Gain(gain_db=master_db),
        Limiter(threshold_db=-0.3, release_ms=50.0),
    ])


def render(audio, sr, params):
    board = build_board(params)
    mono = audio.astype(np.float32)
    processed = board(mono, sr)
    return np.asarray(processed, dtype=np.float32).reshape(-1)
