import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import os



def normalise(filename):
    # Normalise signal
    sampleRate, recording = wave.read(f'recordings/{filename}.wav')

    recording = (recording - np.mean(recording)) / np.std(recording)
    
    wave.write(f'recordings/{filename}_norm.wav', sampleRate, recording)

# normalise('noise_downsampled')


def downSample(filename, rate=16000):
    # Down sample audio files to 16kHz
    sampleRate, recording = wave.read(f'recordings/{filename}.wav')

    if sampleRate != rate:
        N = round((len(recording) * rate)/ sampleRate)
        recording = signal.resample(recording, N)
        
        wave.write(f'recordings/{filename}_{rate}.wav', rate, ref)

# downSample('possum')


def generateRef(filename, name):
    # Save reference spectrogram as .npy file
    sampleRate, ref = wave.read(f'recordings/{filename}.wav')

    fr, tr, Sr = signal.spectrogram(ref, fs=sampleRate)
    # e = 1e-11
    # Sr = np.log(Sr + e)

    np.save(f'reference/original/{name}_NL.npy', Sr)

generateRef('ref/noise_snip', 'noise')

# White noise
# w = np.ones((129,30)) * 100
# np.save(f'reference/normalised/white100.npy', w)