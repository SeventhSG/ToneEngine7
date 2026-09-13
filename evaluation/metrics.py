import numpy as np

from audio.rendering.params import PARAM_NAMES, PARAM_MIN, PARAM_MAX
from audio.similarity.stft_loss import multi_resolution_stft_distance


def parameter_mae(pred_vector, true_vector):
    """Both in normalized [0, 1] space. Returns MAE per param (in original
    0-10 units) plus overall mean.
    """
    scale = PARAM_MAX - PARAM_MIN
    abs_err = np.abs(np.asarray(pred_vector) - np.asarray(true_vector)) * scale
    per_param = {name: float(err) for name, err in zip(PARAM_NAMES, abs_err)}
    per_param["mean"] = float(abs_err.mean())
    return per_param


def audio_similarity(ref_audio, gen_audio):
    return multi_resolution_stft_distance(ref_audio, gen_audio)
