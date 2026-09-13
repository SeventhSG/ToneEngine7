import numpy as np
import soundfile as sf


def save_wav(path, audio, sr):
    sf.write(str(path), audio.astype(np.float32), sr, subtype="FLOAT")


def load_wav(path, sr=None):
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr is not None and file_sr != sr:
        raise ValueError(f"expected sr={sr}, got {file_sr} from {path}")
    return audio, file_sr


def peak_normalize(audio, target_peak=0.95):
    peak = np.max(np.abs(audio)) + 1e-8
    return (audio / peak * target_peak).astype(np.float32)
