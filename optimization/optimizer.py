"""Gradient-free refinement of the tone predictor's initial guess.

The virtual amp (pedalboard) is not differentiable, so this cannot be plain
gradient descent through the renderer. CMA-ES treats it as a black box:
render a candidate, score it against the reference with the same
multi-resolution STFT distance used everywhere else in this project, and
adjust. The CNN's prediction is only the starting point, per INSTRUCTIONS.md.
"""

import cma
import numpy as np

from audio.rendering.params import PARAM_NAMES, denormalize
from audio.rendering.virtual_amp import render as amp_render
from audio.similarity.stft_loss import multi_resolution_stft_distance


def _loss(vector01, source_audio, sr, ref_audio):
    vector01 = np.clip(vector01, 0.0, 1.0)
    params = dict(zip(PARAM_NAMES, denormalize(vector01.tolist())))
    gen_audio = amp_render(source_audio, sr, params)
    result = multi_resolution_stft_distance(ref_audio, gen_audio)
    return result["spectral_convergence"] + result["log_mag_error"]


def optimize(initial_vector01, source_audio, sr, ref_audio, max_evals=300, sigma0=0.15, seed=0):
    """initial_vector01: length-6 array in [0, 1], the CNN's prediction.
    Returns (best_vector01, n_evals_used).
    """
    es = cma.CMAEvolutionStrategy(
        np.asarray(initial_vector01, dtype=float).tolist(),
        sigma0,
        {"bounds": [0.0, 1.0], "maxfevals": max_evals, "seed": seed, "verbose": -9},
    )
    while not es.stop():
        candidates = es.ask()
        losses = [_loss(np.array(c), source_audio, sr, ref_audio) for c in candidates]
        es.tell(candidates, losses)

    best_vector01 = np.clip(np.array(es.result.xbest), 0.0, 1.0)
    return best_vector01, es.result.evaluations
