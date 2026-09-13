"""Synthetic cabinet impulse responses.

Not measurements of real cabinets (no licensed IR library here) -- each is a
deterministic sum of decaying resonant modes standing in for a cabinet's
characteristic peaks and rolloff. Good enough to give Experiment 2 a genuine
"which cabinet was this" classification problem; swapping in measured IRs
later is a drop-in replacement (feed pedalboard.Convolution a real wav
instead of one of these arrays).
"""

import numpy as np

CABINET_NAMES = ["closed_back_4x12", "open_back_2x12", "vintage_1x12", "modern_4x12_bright"]

_SPECS = {
    "closed_back_4x12": {"modes": [(100, 45, 1.0), (1200, 25, 0.5)], "rolloff_hz": 5000},
    "open_back_2x12": {"modes": [(90, 35, 1.0), (2000, 20, 0.6)], "rolloff_hz": 6500},
    "vintage_1x12": {"modes": [(150, 30, 1.0), (900, 22, 0.4)], "rolloff_hz": 3500},
    "modern_4x12_bright": {"modes": [(110, 45, 1.0), (1500, 25, 0.5), (4000, 15, 0.3)], "rolloff_hz": 7500},
}


def _lowpass_by_convolution(signal, sr, cutoff_hz, taps=64):
    t = np.arange(-taps // 2, taps // 2) / sr
    h = np.sinc(2 * cutoff_hz * t)
    h *= np.hanning(len(h))
    h /= h.sum()
    return np.convolve(signal, h, mode="same")


def make_ir(cabinet_name, sr, duration=0.08):
    spec = _SPECS[cabinet_name]
    t = np.arange(int(duration * sr), dtype=np.float64) / sr
    ir = np.zeros_like(t)
    ir[0] = 1.0  # direct impulse component

    for freq, decay_rate, amp in spec["modes"]:
        ir += amp * np.sin(2 * np.pi * freq * t) * np.exp(-decay_rate * t)

    ir = _lowpass_by_convolution(ir, sr, spec["rolloff_hz"])
    ir /= np.max(np.abs(ir)) + 1e-8
    return ir.astype(np.float32)


def all_irs(sr):
    return {name: make_ir(name, sr) for name in CABINET_NAMES}
