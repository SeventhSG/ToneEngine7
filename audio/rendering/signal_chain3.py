"""Experiment 3's renderer: the full Stage 2 rig from INSTRUCTIONS.md.

    noise gate -> compressor -> overdrive -> amp -> EQ pedal (effects loop)
               -> cabinet IR -> microphone

Gate, compressor, overdrive and EQ can each be on or off. The amp voicings
and cabinet IRs are Experiment 2's (audio/rendering/amp_models.py and
cabinets.py); the microphone models are new (mics.py). Everything is
described by EXP3_SPEC, so the dataset, model heads, loss masking and
metrics all follow from it.
"""

import numpy as np
from pedalboard import (Clipping, Compressor, Distortion, Gain, HighShelfFilter, Limiter, LowShelfFilter,
                        NoiseGate, Pedalboard, PeakFilter)

from audio.rendering.amp_models import AMP_NAMES, AMP_SPECS
from audio.rendering.cabinets import CABINET_NAMES
from audio.rendering.chain_spec import ChainSpec
from audio.rendering.mics import MIC_NAMES, mic_plugins
from audio.rendering.signal_chain import _get_convolution

EQ_BANDS_HZ = [100, 400, 800, 1600, 3200]
EQ_KNOBS = [f"eq_{hz}" for hz in EQ_BANDS_HZ]

EXP3_SPEC = ChainSpec(
    continuous=["gate_threshold", "comp_sustain", "comp_level", "od_drive", "od_level",
                "gain", "bass", "mid", "treble", "presence", "master",
                *EQ_KNOBS, "mic_position", "mic_distance"],
    switches={
        "gate_on": ["gate_threshold"],
        "comp_on": ["comp_sustain", "comp_level"],
        "overdrive_on": ["od_drive", "od_level"],
        "eq_on": EQ_KNOBS,
    },
    choices={"amp": AMP_NAMES, "cabinet": CABINET_NAMES, "mic": MIC_NAMES},
)


def _lerp(knob, lo, hi):
    t = min(max(knob / 10.0, 0.0), 1.0)
    return lo + t * (hi - lo)


def build_board(chain, sr):
    plugins = []

    if chain["gate_on"]:
        plugins.append(NoiseGate(threshold_db=_lerp(chain["gate_threshold"], -80.0, -30.0),
                                 ratio=10.0, attack_ms=1.0, release_ms=80.0))

    if chain["comp_on"]:
        sustain = chain["comp_sustain"]
        plugins += [Compressor(threshold_db=_lerp(sustain, -6.0, -40.0), ratio=_lerp(sustain, 2.0, 10.0),
                               attack_ms=5.0, release_ms=100.0),
                    Gain(gain_db=_lerp(chain["comp_level"], -6.0, 12.0))]

    if chain["overdrive_on"]:
        plugins += [Clipping(threshold_db=-_lerp(chain["od_drive"], 0.0, 24.0)),
                    Gain(gain_db=_lerp(chain["od_level"], -6.0, 6.0))]

    amp = AMP_SPECS[AMP_NAMES[chain["amp"]]]
    plugins += [
        Gain(gain_db=_lerp(chain["gain"], -10.0, 30.0)),
        Distortion(drive_db=amp["drive_db"]),
        LowShelfFilter(cutoff_frequency_hz=amp["bass_hz"], gain_db=_lerp(chain["bass"], -12.0, 12.0), q=0.7),
        PeakFilter(cutoff_frequency_hz=amp["mid_hz"], gain_db=_lerp(chain["mid"], -12.0, 12.0), q=0.7),
        HighShelfFilter(cutoff_frequency_hz=amp["treble_hz"], gain_db=_lerp(chain["treble"], -12.0, 12.0), q=0.7),
        HighShelfFilter(cutoff_frequency_hz=amp["presence_hz"], gain_db=_lerp(chain["presence"], 0.0, 15.0), q=0.7),
        Gain(gain_db=_lerp(chain["master"], -24.0, 0.0)),
    ]

    if chain["eq_on"]:
        plugins += [PeakFilter(cutoff_frequency_hz=float(hz), gain_db=_lerp(chain[k], -12.0, 12.0), q=1.0)
                    for hz, k in zip(EQ_BANDS_HZ, EQ_KNOBS)]

    convolution = _get_convolution(CABINET_NAMES[chain["cabinet"]], sr)
    convolution.reset()
    plugins.append(convolution)

    plugins += mic_plugins(MIC_NAMES[chain["mic"]], chain["mic_position"], chain["mic_distance"])
    plugins.append(Limiter(threshold_db=-0.3, release_ms=50.0))
    return Pedalboard(plugins)


def render(audio, sr, chain):
    board = build_board(chain, sr)
    processed = board(np.asarray(audio, dtype=np.float32), sr)
    return np.asarray(processed, dtype=np.float32).reshape(-1)
