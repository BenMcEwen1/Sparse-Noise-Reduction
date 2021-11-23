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


def downSample(recording, sampleRate, rate=16000):
    # Down sample audio files to 16kHz
    if sampleRate != rate:
        N = round((len(recording) * rate)/ sampleRate)
        recording = signal.resample(recording, N)
        print(f'Audio downsampled to {rate}')

    return recording

# downSample('possum')


def generateRef(filename, name):
    # Save reference spectrogram as .npy file
    sampleRate, ref = wave.read(f'recordings/{filename}.wav')

    # ref = downSample(ref, sampleRate)

    fr, tr, Sr = signal.spectrogram(ref, fs=sampleRate)

    np.save(f'reference/new/{name}.npy', Sr)
    print('Reference saved')

generateRef('ref/new/possum_snip', 'possum')

# White noise
# w = np.ones((129,30)) * 100
# np.save(f'reference/normalised/white100.npy', w)

# l = np.load('./reference/new/screech.npy')
# print(l.shape)