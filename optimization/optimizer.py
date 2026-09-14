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


def _cma_seed(seed):
    # cma treats a seed of 0 (or None) as "seed from the clock", so seed=0
    # silently gave a different search every run. Shift by one so every
    # seed, including the default 0, is a real fixed seed.
    return int(seed) + 1


def optimize(initial_vector01, source_audio, sr, ref_audio, max_evals=300, sigma0=0.15, seed=0):
    """initial_vector01: length-6 array in [0, 1], the CNN's prediction.
    Returns (best_vector01, n_evals_used).
    """
    loss_fn = lambda vector01: _loss(vector01, source_audio, sr, ref_audio)
    return optimize_generic(initial_vector01, loss_fn, max_evals=max_evals, sigma0=sigma0, seed=seed)


def optimize_generic(initial_vector01, loss_fn, max_evals=300, sigma0=0.15, seed=0):
    """Same CMA-ES loop, but for any continuous vector in [0, 1] and any
    black-box loss_fn(vector01) -> float. Used when the parameter meaning
    (which knobs, which renderer) varies by experiment, e.g. Experiment 2's
    continuous knobs with the discrete amp/cabinet/overdrive choice fixed.
    """
    es = cma.CMAEvolutionStrategy(
        np.asarray(initial_vector01, dtype=float).tolist(),
        sigma0,
        {"bounds": [0.0, 1.0], "maxfevals": max_evals, "seed": _cma_seed(seed), "verbose": -9},
    )
    while not es.stop():
        candidates = es.ask()
        losses = [loss_fn(np.clip(np.array(c), 0.0, 1.0)) for c in candidates]
        es.tell(candidates, losses)

    best_vector01 = np.clip(np.array(es.result.xbest), 0.0, 1.0)
    return best_vector01, es.result.evaluations


class _ResumableSearch:
    """One CMA-ES run that can be advanced in slices of budget, so several
    runs can be compared and the losers dropped partway through. Only whole
    generations are evaluated, and never more than the slice allows. The
    starting point itself is scored, so the result is never worse than it.
    """

    def __init__(self, x0, loss_fn, sigma0, seed):
        x0 = np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)
        self.loss_fn = loss_fn
        self.es = cma.CMAEvolutionStrategy(
            x0.tolist(), sigma0, {"bounds": [0.0, 1.0], "seed": _cma_seed(seed), "verbose": -9},
        )
        self.best_x = x0
        self.best_loss = loss_fn(x0)
        self.n_evals = 1

    def advance(self, n_evals):
        popsize = self.es.popsize
        used = 0
        while used + popsize <= n_evals and not self.es.stop():
            candidates = [np.clip(np.array(c), 0.0, 1.0) for c in self.es.ask()]
            losses = [self.loss_fn(c) for c in candidates]
            self.es.tell([c.tolist() for c in candidates], losses)
            used += popsize
            i = int(np.argmin(losses))
            if losses[i] < self.best_loss:
                self.best_loss, self.best_x = losses[i], candidates[i]
        self.n_evals += used
        return used


def optimize_with_choices(candidates, loss_fn, max_evals=1800, keep=1,
                          screen_evals=300, mid_evals=0, sigma0=0.15, seed=0):
    """Search over discrete choices (which amp, which cabinet, overdrive on or
    off) as well as the continuous knobs, within one total render budget.

    candidates: list of (choice, initial_vector01), best guess first, where
        choice is whatever loss_fn needs to know which chain to render.
    loss_fn(choice, vector01) -> float

    Successive halving: every candidate gets a short CMA-ES run, the `keep`
    best by loss so far get a longer one, and the single best then spends
    whatever budget is left. Candidates are never mixed, so a run's
    covariance only ever describes the knobs of one fixed chain.

    The defaults are the schedule that won on Experiment 2's validation set:
    below roughly 300 renders per candidate, the short runs rank chains by
    how close each one happened to start, not by how close it can get, and
    the search loses to simply trusting the CNN's choice.

    Returns (best_choice, best_vector01, n_evals_used, trace), where trace
    records each candidate's best loss at every round for later analysis.
    """
    if len(candidates) * screen_evals > max_evals:
        raise ValueError(f"{len(candidates)} candidates x {screen_evals} screening renders "
                         f"exceeds the budget of {max_evals}")
    searches = []
    for choice, x0 in candidates:
        searches.append((choice, _ResumableSearch(x0, lambda v, c=choice: loss_fn(c, v), sigma0, seed)))
    for _, search in searches:
        search.advance(screen_evals - 1)

    def total():
        return sum(s.n_evals for _, s in searches)

    trace = {"screen": [(choice, s.best_loss) for choice, s in searches]}

    survivors = sorted(searches, key=lambda cs: cs[1].best_loss)[:keep]
    for _, search in survivors:
        search.advance(min(mid_evals, max_evals - total()))
    trace["mid"] = [(choice, s.best_loss) for choice, s in survivors]

    choice, winner = min(survivors, key=lambda cs: cs[1].best_loss)
    winner.advance(max_evals - total())
    return choice, winner.best_x, total(), trace
