import torch
import torchaudio


class LogMelFeature:
    def __init__(self, sr, n_fft=1024, hop_length=256, n_mels=64):
        self.sr = sr
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def __call__(self, audio):
        """audio: 1D numpy array or torch tensor. Returns [1, n_mels, time] tensor."""
        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        mel = self.transform(audio)
        log_mel = torch.log(mel + 1e-6)
        return log_mel.unsqueeze(0)
