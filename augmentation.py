import numpy as np
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import random
from spectral_noise_grating import spectrogram, autoThreshold, mask, reconstruct, plot_spectrogram, ISTFT, convertToAmp
import librosa
import os
import soundfile as sf

def addNoise(sampleRate, recording):
    # Add guassian noise
    n = 2*np.std(recording)
    noise = np.random.normal(-n,n, len(recording))
    ref = np.add(recording, noise)
    return ref


def superimpose(recording, vocalisation):
    # Overlay vocalisations with "empty" field recordings
    index = random.randint(0,len(recording))

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


def load(directory):
    recording, sampleRate = librosa.load(directory, sr=None)
    return recording, sampleRate


def select(directory):
    # Select empty field recordings
    empty = []
    for dirpaths, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            empty.append(f"{directory}/{filename}")

    return empty


def generate(vocalisation):
    # Generate synthetic data for a known vocalisation
    directory = './recordings/empty'
    empty = select(directory)

    # Denoise vocalisation
    denoised = denoise(vocalisation)

    # Superimpose vocalisation at different times 
    for i,e in enumerate(empty):
        print(e)
        field, sampleRate = load(e)
        synthetic = superimpose(field, denoised)
        sf.write(f'./training/denoised{i}.wav', synthetic, sampleRate)


vocalisation = './recordings/downsampled/possum16k.wav'
generate(vocalisation)

