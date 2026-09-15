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


def synth_phrase(sr, duration, rng, min_notes=1, max_notes=3):
    """Like synth_source, but notes start at random times and may be damped
    early, so the clip has stretches where nothing is played. synth_source
    strikes every note at t=0 and lets it ring for the whole clip, which
    leaves a noise gate nothing to act on. Kept separate so Experiments 1 and
    2 (which use synth_source) are unchanged.
    """
    n_samples = int(duration * sr)
    n_notes = int(rng.integers(min_notes, max_notes + 1))
    audio = np.zeros(n_samples, dtype=np.float32)
    events = []
    for _ in range(n_notes):
        midi = int(rng.integers(GUITAR_MIDI_LOW, GUITAR_MIDI_HIGH + 1))
        onset = float(rng.uniform(0.0, 0.45 * duration))
        # about half the notes are damped (palm mute / released), the rest ring out
        length = float(rng.uniform(0.08, 0.35)) if rng.uniform() < 0.5 else duration
        note = _pluck(midi_to_freq(midi), duration, sr, rng)
        start = int(onset * sr)
        stop = min(n_samples, start + int(length * sr))
        seg = note[: stop - start].copy()
        fade = min(len(seg), int(0.015 * sr))  # quick damping instead of a click
        if stop < n_samples and fade > 0:
            seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        audio[start:stop] += rng.uniform(0.6, 1.0) * seg
        events.append({"midi": midi, "onset_s": onset, "length_s": min(length, duration - onset)})

    peak = np.max(np.abs(audio)) + 1e-8
    audio = (audio / peak * 0.8).astype(np.float32)
    return audio, {"notes": events}


def _lerp(v01, lo, hi):
    return lo + (min(max(v01, 0.0), 10.0) / 10.0) * (hi - lo)


def _pluck_source(freq, duration, sr, rng, pickup01, dynamics01, n_harmonics=10):
    """Like _pluck, but brightness/attack come from `dynamics01` (harder pick
    = brighter, louder transient) instead of being drawn randomly, and each
    harmonic is scaled by a pickup-position comb filter: a pickup at
    fraction `pos_frac` of the string length attenuates the harmonics whose
    node falls there (the classic bridge-vs-neck-pickup brightness
    difference), abs(sin(k * pi * pos_frac)).
    """
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    signal = np.zeros_like(t)

    brightness = _lerp(dynamics01, 0.4, 1.0)
    pos_frac = _lerp(pickup01, 0.08, 0.42)
    base_decay = rng.uniform(2.5, 6.0)
    phase = rng.uniform(0, 2 * np.pi, size=n_harmonics)

    for k in range(1, n_harmonics + 1):
        amp = (1.0 / k) ** (1.0 + (1.0 - brightness) * 1.5)
        amp *= abs(np.sin(k * np.pi * pos_frac))
        decay_rate = base_decay * (1.0 + 0.15 * k)
        signal += amp * np.sin(2 * np.pi * k * freq * t + phase[k - 1]) * np.exp(-decay_rate * t)

    attack_len = max(1, int(0.004 * sr))
    noise = rng.normal(0, 1, size=attack_len).astype(np.float32)
    noise *= np.exp(-np.arange(attack_len) / (attack_len * 0.3))
    attack_gain = _lerp(dynamics01, 0.15, 0.7)
    signal[:attack_len] += attack_gain * noise

    peak = np.max(np.abs(signal)) + 1e-8
    return (signal / peak).astype(np.float32)


def synth_source_pickup(sr, duration, rng, pickup01, dynamics01, tuning01, min_notes=1, max_notes=2):
    """Like synth_source, but pickup position, playing dynamics and tuning
    are explicit, recorded parameters (Experiment 4's source/performance
    variables) instead of the random per-note draws synth_source uses.
    Returns (audio, note_events).
    """
    tuning_semitones = _lerp(tuning01, -4.0, 2.0)
    n_notes = int(rng.integers(min_notes, max_notes + 1))
    midi_notes = [int(rng.integers(GUITAR_MIDI_LOW, GUITAR_MIDI_HIGH + 1)) + tuning_semitones for _ in range(n_notes)]

    audio = np.zeros(int(duration * sr), dtype=np.float32)
    for midi in midi_notes:
        freq = midi_to_freq(midi)
        note_audio = _pluck_source(freq, duration, sr, rng, pickup01, dynamics01)
        gain = _lerp(dynamics01, 0.5, 1.0)
        audio += gain * note_audio

    peak = np.max(np.abs(audio)) + 1e-8
    audio = (audio / peak * 0.8).astype(np.float32)

    note_events = {"midi_notes": midi_notes, "tuning_semitones": tuning_semitones}
    return audio, note_events


def add_noise_floor(audio, sr, rng, level_db_range=(-70.0, -40.0)):
    """A real DI signal is never silent: pickups hum and the interface hisses.
    Adds white hiss plus mains hum (50 or 60 Hz with a few harmonics) at a
    random level in dBFS, so a noise gate has something to do. The level is a
    property of the source, not the signal chain, and is returned so it can be
    recorded rather than predicted.
    """
    level_db = float(rng.uniform(*level_db_range))
    t = np.arange(len(audio), dtype=np.float32) / sr
    mains = 50.0 if rng.uniform() < 0.5 else 60.0
    hum = sum(np.sin(2 * np.pi * mains * k * t + rng.uniform(0, 2 * np.pi)) / k for k in (1, 2, 3))
    hiss = rng.normal(0.0, 1.0, size=len(audio))
    noise = 0.5 * hiss / (np.std(hiss) + 1e-8) + 0.5 * hum / (np.std(hum) + 1e-8)
    noise *= 10.0 ** (level_db / 20.0)
    return (audio + noise).astype(np.float32), level_db

