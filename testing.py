from scipy.stats import entropy, kstest, uniform
from scipy.signal import normalize
import numpy as np
import random
import scipy.io.wavfile as wave
from scipy import signal
import matplotlib.pyplot as plt

filename = 'downsampled/cat16k'
sampleRate, s = wave.read(f'recordings/{filename}.wav')

fp, tp, Sp = signal.spectrogram(s, fs=sampleRate)


possum = np.load('reference/original/possum_NL.npy')
cat = np.load('reference/original/cat_NL.npy')



print(f"Sum Energy: {possum.sum()}")

# Smoothing to reduce the effects of noise
kernel = np.ones((2,2)) * 0.5
# smoothed = signal.convolve2d(possum, kernel, mode='same', boundary='wrap', fillvalue=0)



def normalise(mask):
    # Normalise to prevent higher energy masks becoming biased
    mask = signal.convolve2d(mask, kernel, mode='same', boundary='wrap', fillvalue=0)
    norm = np.linalg.norm(mask)
    mask = np.divide(mask, norm)
    mask = mask / mask.sum()
    return mask


possum = normalise(possum)
cat = normalise(cat)


plt.imshow(possum)
plt.show()

plt.imshow(cat)
plt.show()

filters = [possum, cat]

lower = 0
upper = 0
cor = []

for mask in filters:
    c = signal.correlate(Sp, mask, 'valid')

    cor.append(c[0])

    if c.min() < lower:
        lower = c.min()
    if c.max() > upper:
        upper = c.max()

scaled = []
for c in cor:
    c = np.interp(c, (lower,upper), (0,1)) 
    scaled.append(c)


for s in scaled:
    plt.plot(s)
    plt.ylim(0,1.1)
    plt.show()