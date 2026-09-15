"""Synthetic microphone models for the recording chain (Experiment 3).

Not measurements of real microphones. Each type is a small filter shape
standing in for a family's typical character, and two placement knobs
stand in for how a mic sounds on a cabinet:

    position  0 = centre of the speaker cone (brightest) .. 10 = edge (darker)
    distance  0 = touching the grille (proximity-effect bass boost)
              .. 10 = back from the cab (less bass, plus an early reflection
              that comb-filters the response the way a nearby surface does)

The project renders at 16 kHz, so everything here lives below 8 kHz.
"""

from pedalboard import Delay, HighShelfFilter, HighpassFilter, LowShelfFilter, PeakFilter

MIC_NAMES = ["dynamic", "condenser", "ribbon"]

# type-specific voicing, applied before placement
_VOICING = {
    # tight lows, forward upper mids: the classic close-miked guitar sound
    "dynamic": lambda: [HighpassFilter(cutoff_frequency_hz=90.0),
                        PeakFilter(cutoff_frequency_hz=4000.0, gain_db=5.0, q=1.2)],
    # flat and extended, slight air
    "condenser": lambda: [HighpassFilter(cutoff_frequency_hz=40.0),
                          HighShelfFilter(cutoff_frequency_hz=5000.0, gain_db=2.0, q=0.7)],
    # dark and smooth: rolled-off top, a little low-end weight
    "ribbon": lambda: [HighShelfFilter(cutoff_frequency_hz=2500.0, gain_db=-8.0, q=0.7),
                       LowShelfFilter(cutoff_frequency_hz=200.0, gain_db=2.0, q=0.7)],
}


def _lerp(knob, lo, hi):
    t = min(max(knob / 10.0, 0.0), 1.0)
    return lo + t * (hi - lo)


def mic_plugins(mic_name, position, distance):
    plugins = _VOICING[mic_name]()
    # off-axis toward the cone edge loses treble
    plugins.append(HighShelfFilter(cutoff_frequency_hz=2500.0, gain_db=_lerp(position, 3.0, -9.0), q=0.7))
    # proximity effect: strong bass boost up close, slightly thin further back
    plugins.append(LowShelfFilter(cutoff_frequency_hz=150.0, gain_db=_lerp(distance, 8.0, -3.0), q=0.7))
    # an early reflection that grows with distance
    plugins.append(Delay(delay_seconds=_lerp(distance, 0.001, 0.006), feedback=0.0, mix=_lerp(distance, 0.0, 0.3)))
    return plugins
