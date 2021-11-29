from spectral_noise_grating import spectrogram, autoThreshold, mask, reconstruct, plot_spectrogram
import librosa
import numpy as np
import matplotlib.pyplot as plt


recording, sample_rate = librosa.load('./recordings/downsampled/possum16k.wav', sr=None)

sig_stft, sig_stft_db = spectrogram(recording)

thresh = autoThreshold(sig_stft_db)

masked, mask = mask(thresh, sig_stft, sig_stft_db)

reconstructed, original = reconstruct(masked, recording)

# Original and denoised
plot_spectrogram(original, title="Original spectrogram")
plot_spectrogram(reconstructed, title="Reconstructed spectrogram")
