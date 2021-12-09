# Generate data augmentation and synthetic data for a given field recording

import numpy as np
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import random
from spectral_noise_grating import spectrogram, autoThreshold, mask, reconstruct, plot_spectrogram, ISTFT, convertToAmp
import librosa

directory = './recordings/downsampled/field16k'
sampleRate, recording = wave.read(f'{directory}.wav')
_, vocalisation = wave.read('./recordings/downsampled/possum16k.wav')

def addNoise(sampleRate, recording):
    # Add guassian noise
    n = 2*np.std(recording)
    noise = np.random.normal(-n,n, len(recording))
    ref = np.add(recording, noise)
    return ref

def timeShift():
    # Add time shift
    pass

def synthetic(recording, vocalisation):
    # Overlay vocalisations with "empty" field recordings
    # combine = 
    index = 2000000

    # index = random.randint(0,len(recording))

    padded = np.zeros(len(recording))
    padded = np.insert(padded, index, vocalisation)

    padded = padded[0:len(recording)]

    combined = np.add(recording, padded)
    return combined


def denoise(directory):
    # Denoise recording 
    vocalisation, sample_rate = librosa.load(directory, sr=None)

    sig_stft, sig_stft_db = spectrogram(vocalisation)

    thresh = autoThreshold(sig_stft_db)

    masked, _ = mask(thresh, sig_stft, sig_stft_db)

    reconstructed, original = reconstruct(masked, vocalisation)

    denoised = convertToAmp(reconstructed)
    denoised = ISTFT(denoised)

    return denoised

# Denoised vocalisation
directory = './recordings/downsampled/possum16k.wav'
vocalisation = denoise(directory)

# Denoised field recording
directory = './recordings/downsampled/field16k.wav'
recording = denoise(directory)

# Generate synthetic
c = synthetic(recording, vocalisation)

plt.specgram(c)
plt.show()


# ref = addNoise(sampleRate, recording)

# plt.figure(1)
# plt.specgram(ref, Fs=sampleRate)

# plt.figure(2)
# plt.specgram(recording, Fs=sampleRate)
# plt.show()

