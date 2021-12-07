# Generate data augmentation and synthetic data for a given field recording

import numpy as np
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt

directory = './recordings/downsampled/field16k'
sampleRate, recording = wave.read(f'{directory}.wav')

def addNoise(sampleRate, recording):
    # Add guassian noise
    n = 2*np.std(recording)
    noise = np.random.normal(-n,n, len(recording))
    ref = np.add(recording, noise)
    return ref

def timeShift():
    # Add time shift
    pass

def synthetic():
    # Overlay vocalisations with "empty" field recordings


ref = addNoise(sampleRate, recording)

plt.figure(1)
plt.specgram(ref, Fs=sampleRate)

plt.figure(2)
plt.specgram(recording, Fs=sampleRate)
plt.show()

