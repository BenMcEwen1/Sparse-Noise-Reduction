import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import os
from segment import extract, correlation
from spectral_noise_grating import spectrogram, autoThreshold, mask, reconstruct, ISTFT, plot_spectrogram
import librosa
import soundfile as sf


filename = 'downsampled/field16k'
directory = './reference/new'

# Denoise
recording, sample_rate = librosa.load(f'./recordings/{filename}.wav', sr=None)

print(sample_rate)

sig_stft, sig_stft_db = spectrogram(recording)
thresh = autoThreshold(sig_stft_db)
masked, mask = mask(thresh, sig_stft, sig_stft_db)
reconstructed, original = reconstruct(masked, recording)

denoised = ISTFT(reconstructed)
# plot_spectrogram(reconstructed, 'denoised')
# denoised = np.asarray(denoised, dtype='int16')
# wave.write('./segmented/denoised.wav', sample_rate, denoised)

sf.write('./segmented/denoised.wav', denoised, sample_rate)

# plt.imshow(np.flipud(original))
# plt.show()


# # Segement field recording
# masks, calls = extract(directory)

# def normalise(mask):
#     # Normalise to prevent higher energy masks becoming biased
#     norm = np.linalg.norm(mask)
#     mask = np.divide(mask, norm)
#     mask = mask / mask.sum()
#     return mask

# def correlation(Sp, masks, sampleRate):
#     # Convolve spectrogram with ref to generate correlation
#     Sp = np.flipud(Sp)
#     # Normalisation

#     kernel = np.ones((2,2)) * 0.5

#     cor = []
#     scaled = []

#     lower = 0
#     upper = 0

#     for mask in masks:
#         # Normalise Mask
#         mask = normalise(mask)

#         # Smoothing (Optional)
#         mask = signal.convolve2d(mask, kernel, mode='same', boundary='wrap', fillvalue=0)

#         c = signal.correlate(Sp, mask, mode="valid")

#         if c.min() < lower:
#             lower = c.min()
#         if c.max() > upper:
#             upper = c.max()
        
#         cor.append(c[0])

#     # Scale correlation relative to upper and lower values
#     for c in cor:
#         c = np.interp(c, (lower,upper), (0,1)) 
#         scaled.append(c)

#     return scaled

# scaled = correlation(reconstructed, masks, sample_rate)

# for i,s in enumerate(scaled):
#     print(calls[i])
#     plt.plot((s))
#     plt.ylim([0,1.1])
#     plt.show()