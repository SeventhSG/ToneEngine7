"""Multi-resolution STFT distance between two waveforms.

Raw waveform MSE is a poor proxy for tonal similarity (a few samples of
phase shift can look completely different in the time domain while
sounding identical). Comparing log-magnitude spectra at several FFT
resolutions is the standard fix used across neural vocoder and amp-sim
literature, so that's what this reports.
"""

import torch


FFT_SIZES = [512, 1024, 2048]
HOP_DIVISOR = 4


def _stft_mag(audio, n_fft):
    hop_length = n_fft // HOP_DIVISOR
    window = torch.hann_window(n_fft, device=audio.device)
    spec = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length, window=window,
        return_complex=True, center=True,
    )
    return torch.abs(spec) + 1e-7


def multi_resolution_stft_distance(ref, gen):
    """ref, gen: 1D torch tensors, same length (or gen padded/truncated to match).
    Returns a dict with spectral convergence and log-magnitude L1 error,
    averaged across resolutions, plus a single scalar similarity in (0, 1].
    """
    if not torch.is_tensor(ref):
        ref = torch.as_tensor(ref, dtype=torch.float32)
    if not torch.is_tensor(gen):
        gen = torch.as_tensor(gen, dtype=torch.float32)

    n = min(ref.shape[-1], gen.shape[-1])
    ref = ref[..., :n]
    gen = gen[..., :n]

    sc_total = 0.0
    mag_total = 0.0
    for n_fft in FFT_SIZES:
        ref_mag = _stft_mag(ref, n_fft)
        gen_mag = _stft_mag(gen, n_fft)

        sc = torch.norm(ref_mag - gen_mag, p="fro") / torch.norm(ref_mag, p="fro")
        mag = torch.mean(torch.abs(torch.log(ref_mag) - torch.log(gen_mag)))

        sc_total += sc.item()
        mag_total += mag.item()

    n_res = len(FFT_SIZES)
    spectral_convergence = sc_total / n_res
    log_mag_error = mag_total / n_res
    similarity = 1.0 / (1.0 + spectral_convergence + log_mag_error)

    return {
        "spectral_convergence": spectral_convergence,
        "log_mag_error": log_mag_error,
        "similarity": similarity,
    }
