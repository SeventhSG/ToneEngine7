"""Loss, decoding and metrics shared by Experiment 3's training, evaluation
and optimization scripts. Everything is driven by the chain spec."""

import numpy as np
import torch
import torch.nn.functional as F

from audio.rendering.chain_spec import PARAM_MAX, PARAM_MIN, denormalize
from audio.rendering.signal_chain3 import EXP3_SPEC as SPEC


def compute_loss(pred, target):
    """Masked MSE on the knobs (knobs of pedals that are off do not count),
    plus binary cross-entropy per switch and cross-entropy per choice, summed
    so each head carries weight as in Experiment 2."""
    mask = target["mask"]
    continuous = (((pred["continuous"] - target["continuous"]) ** 2) * mask).sum() / mask.sum().clamp(min=1)
    switch = F.binary_cross_entropy_with_logits(pred["switch_logits"], target["switches"], reduction="none").mean(0).sum()
    choice = sum(F.cross_entropy(logits, target["choices"][:, j]) for j, logits in enumerate(pred["choice_logits"]))
    return continuous + switch + choice


def decode(pred, i):
    """One example's prediction as (knobs on the 0-1 scale, switches, choices)."""
    knobs01 = pred["continuous"][i].detach().cpu().numpy()
    switches = (pred["switch_logits"][i] > 0).cpu().numpy().astype(np.int8)
    choices = np.array([int(logits[i].argmax()) for logits in pred["choice_logits"]], dtype=np.int8)
    return knobs01, switches, choices


def chain_from(knobs01, switches, choices):
    return SPEC.from_arrays(denormalize(knobs01), switches, choices)


def knob_mae(knobs01, true01, true_switches):
    """Mean absolute knob error on the 0-10 scale over the knobs that matter
    in the true chain (a pedal that was off has no correct knob setting)."""
    mask = SPEC.knob_mask(true_switches)
    err = np.abs(np.asarray(knobs01) - np.asarray(true01)) * (PARAM_MAX - PARAM_MIN)
    return float((err * mask).sum() / max(mask.sum(), 1.0))


def per_knob_errors(knobs01, true01, true_switches):
    """Per-knob absolute errors (0-10 scale), NaN where the knob does not matter."""
    mask = SPEC.knob_mask(true_switches)
    err = np.abs(np.asarray(knobs01) - np.asarray(true01)) * (PARAM_MAX - PARAM_MIN)
    return np.where(mask > 0, err, np.nan)
