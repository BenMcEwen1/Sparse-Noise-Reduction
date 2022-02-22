import numpy as np
from sklearn.decomposition import NMF
from scipy import signal
import scipy.io.wavfile as wave

X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])

recording = "./recordings/downsampled/possum16k.wav"

sampleRate, s = wave.read(recording)
fp, tp, Sp = signal.spectrogram(s, fs=sampleRate)



model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(Sp)
H = model.components_

X_new = np.array([[1, 0], [1, 6.1], [1, 0], [1, 4], [3.2, 1], [0, 4]])
W_new = model.transform(X_new)

print(X)
print(W_new)