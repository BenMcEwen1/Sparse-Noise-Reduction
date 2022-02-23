import numpy as np
from sklearn.decomposition import NMF
from scipy import signal
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
from spectral_noise_grating import STFT
import librosa
import librosa.display
import soundfile as sf


# recording = "./recordings/downsampled/cat16k.wav"
recording = "./denoised/denoised.wav"

rec, sample_rate = librosa.load(recording, sr=None)

S = librosa.stft(rec)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

# Non-negative Matrix Factorisation (NMF)
x, phase = librosa.magphase(S)
n_components = 2
W, H = librosa.decompose.decompose(x, n_components=n_components, sort=True, max_iter=1000)


# Reconstruct original
D_k = W.dot(H)
y_k = librosa.istft(D_k * phase)
sf.write('./denoised/reconstruct.wav', y_k, sample_rate)

# Reconstruct components
for n in range(n_components):
    D_k = np.multiply.outer(W[:, n], H[n])
    y_k = librosa.istft(D_k * phase)
    sf.write(f'./denoised/component_{n}.wav', y_k, sample_rate)
