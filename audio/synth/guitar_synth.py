"""Fast synthetic 'DI' guitar source signal.

Not a physical string model. It's an additive pluck (decaying harmonics
plus a short noise attack) that is fully vectorized in numpy, so generating
tens of thousands of source clips is cheap. Good enough to prove the
amp-parameter-recovery concept; swapping in Karplus-Strong or real DI
recordings later is a drop-in replacement for `synth_source`.
"""

import numpy as np

GUITAR_MIDI_LOW = 40   # E2
GUITAR_MIDI_HIGH = 76  # E5


def midi_to_freq(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _pluck(freq, duration, sr, rng, n_harmonics=10):
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    signal = np.zeros_like(t)

    brightness = rng.uniform(0.5, 1.0)
    base_decay = rng.uniform(2.5, 6.0)
    phase = rng.uniform(0, 2 * np.pi, size=n_harmonics)

    for k in range(1, n_harmonics + 1):
        amp = (1.0 / k) ** (1.0 + (1.0 - brightness) * 1.5)
        decay_rate = base_decay * (1.0 + 0.15 * k)
        signal += amp * np.sin(2 * np.pi * k * freq * t + phase[k - 1]) * np.exp(-decay_rate * t)

    attack_len = max(1, int(0.004 * sr))
    noise = rng.normal(0, 1, size=attack_len).astype(np.float32)
    noise *= np.exp(-np.arange(attack_len) / (attack_len * 0.3))
    signal[:attack_len] += 0.4 * noise

    peak = np.max(np.abs(signal)) + 1e-8
    return (signal / peak).astype(np.float32)


def synth_source(sr, duration, rng, min_notes=1, max_notes=3):
    """Returns (audio, note_events) where note_events records what was played,
    for reproducibility.
    """
    n_notes = int(rng.integers(min_notes, max_notes + 1))
    midi_notes = [int(rng.integers(GUITAR_MIDI_LOW, GUITAR_MIDI_HIGH + 1)) for _ in range(n_notes)]

    audio = np.zeros(int(duration * sr), dtype=np.float32)
    for midi in midi_notes:
        freq = midi_to_freq(midi)
        note_audio = _pluck(freq, duration, sr, rng)
        gain = rng.uniform(0.6, 1.0)
        audio += gain * note_audio

    peak = np.max(np.abs(audio)) + 1e-8
    audio = (audio / peak * 0.8).astype(np.float32)

    note_events = {"midi_notes": midi_notes}
    return audio, note_events
