import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_waveform_and_spectrogram_comparison(ref_audio, gen_audio, sr, out_path, title=""):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].plot(ref_audio, linewidth=0.5)
    axes[0, 0].set_title("reference waveform")
    axes[0, 1].plot(gen_audio, linewidth=0.5)
    axes[0, 1].set_title("reconstructed waveform")

    n_fft = 1024
    ref_spec = np.log(np.abs(np.fft.rfft(ref_audio[: n_fft * (len(ref_audio) // n_fft)].reshape(-1, n_fft))).T + 1e-6)
    gen_spec = np.log(np.abs(np.fft.rfft(gen_audio[: n_fft * (len(gen_audio) // n_fft)].reshape(-1, n_fft))).T + 1e-6)

    axes[1, 0].imshow(ref_spec, aspect="auto", origin="lower", cmap="magma")
    axes[1, 0].set_title("reference log-spectrogram")
    axes[1, 1].imshow(gen_spec, aspect="auto", origin="lower", cmap="magma")
    axes[1, 1].set_title("reconstructed log-spectrogram")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
