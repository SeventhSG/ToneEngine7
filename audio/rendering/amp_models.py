"""Three amp voicings for Experiment 2.

Same six knobs as Experiment 1 (gain, bass, mid, treble, presence, master),
but each voicing uses a different fixed distortion character and tone-stack
center frequencies, so the audio actually carries information about which
amp was used, not just where its knobs were set.
"""

AMP_NAMES = ["clean_american", "british_hi_gain", "boutique_low_gain"]

AMP_SPECS = {
    "clean_american": {"drive_db": 8.0, "bass_hz": 150.0, "mid_hz": 800.0, "treble_hz": 3000.0, "presence_hz": 5500.0},
    "british_hi_gain": {"drive_db": 22.0, "bass_hz": 100.0, "mid_hz": 650.0, "treble_hz": 2500.0, "presence_hz": 4500.0},
    "boutique_low_gain": {"drive_db": 4.0, "bass_hz": 200.0, "mid_hz": 1000.0, "treble_hz": 3500.0, "presence_hz": 6000.0},
}
