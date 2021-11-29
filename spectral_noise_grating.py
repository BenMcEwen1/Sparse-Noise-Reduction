# Ref: https://timsainburg.com/noise-reduction-python.html

import scipy.io.wavfile as wave
from numpy.fft import fft 
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.signal import spectrogram, istft, fftconvolve

recording, sample_rate = librosa.load('./recordings/downsampled/field16k.wav', sr=None)
noise, sample_rate = librosa.load('./recordings/downsampled/noise16k.wav', sr=None)

def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def STFT(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def convertToDB(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

def convertToAmp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)

def ISTFT(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


# STFT over signal
n_fft = 2048
win_length = 2048
hop_length = 512


def spectrogram(signal):
    # Generate spectrogram and convert to db
    stft = STFT(signal, n_fft, hop_length, win_length)
    stft_db = convertToDB(abs(stft))

    return stft, stft_db


def threshold(noise_stft_db, n=2):
    # Calculate statistics of noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n 

    # Reshape and extend mask across full recording
    reshaped = np.reshape(noise_thresh, (1,len(noise_thresh)))
    repeats = np.shape(sig_stft_db)[1]

    thresh = np.repeat(reshaped, repeats, axis=0).T

    return thresh


def mask(db_thresh, sig_stft, sig_stft_db):
    # Smoothing filter and normalise
    smooth = np.ones((5,9)) * 0.5
    smooth = smooth / np.sum(smooth)

    # mask if the signal is above the threshold
    mask = sig_stft_db < db_thresh

    # convolve the mask with a smoothing filter
    mask = fftconvolve(mask, smooth, mode="same")
    mask = mask * 1.0

    # mask the signal
    gain = np.min(convertToDB(np.abs(sig_stft)))

    masked = (sig_stft_db * (1 - mask) +  gain * mask)

    return masked, mask


def reconstruct(masked, recoridng):
    reconstructed = convertToDB(abs(convertToAmp(masked)))
    original = convertToDB(abs(STFT(recording, n_fft, hop_length, win_length)))

    return reconstructed, original



noise_stft, noise_stft_db = spectrogram(noise)
sig_stft, sig_stft_db = spectrogram(recording)

db_thresh = threshold(noise_stft_db)

masked, mask = mask(db_thresh, sig_stft, sig_stft_db)

reconstructed, original = reconstruct(masked, recording)

# Original and denoised
plot_spectrogram(original, title="Original spectrogram")
plot_spectrogram(reconstructed, title="Reconstructed spectrogram")