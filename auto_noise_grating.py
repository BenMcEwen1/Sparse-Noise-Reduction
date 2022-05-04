from spectral_noise_grating import spectrogram, autoThreshold, mask, reconstruct, plot_spectrogram, ISTFT, convertToAmp
import librosa
# import numpy as np
# import matplotlib.pyplot as plt
import soundfile as sf
import time

filename = 'lsk_kaka_5min'

recording, sample_rate = librosa.load(f'./Sound Files/{filename}.wav', sr=None)


sig_stft, sig_stft_db = spectrogram(recording)

start = time.time()
thresh = autoThreshold(sig_stft_db)

masked, mask = mask(thresh, sig_stft, sig_stft_db)

reconstructed, original = reconstruct(masked, recording)

denoised = convertToAmp(reconstructed)
denoised = ISTFT(denoised)
stop = time.time()

diff = stop - start
print(f"Run time {diff}")

sf.write(f'./processed/{filename}.wav', recording, sample_rate)
sf.write(f'./processed/{filename}_denoised.wav', denoised, sample_rate)

# Original and denoised
# plot_spectrogram(original, title="Original spectrogram")
# plot_spectrogram(reconstructed, title="Reconstructed spectrogram")