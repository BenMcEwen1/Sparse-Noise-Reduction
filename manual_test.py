import pywt
import numpy as np
from scipy.signal import chirp, convolve
import matplotlib.pyplot as plt
from math import ceil


def pad(signal, level, mode='other'):
    # Pad signal
    l = int(len(signal)/level)
    if (l % 2) != 0:
        l += 1
        l = level*l - len(signal)

        if mode == "circular":
            signal = np.append(signal,signal[0:l])
        elif mode == "fixed":
            signal = np.append(signal,[signal[-1]]*l)
        else: 
            signal = np.append(signal,[0]*l)

    return signal


def unpad(padded, signal):
    # Remove padding
    return padded[0:len(signal)]


def dwt(signal, wavelet):
    # Decompose into approx/detailed coeffs
    w = pywt.Wavelet(wavelet)
    lowPass = w.dec_lo
    highPass = w.dec_hi

    L = len(lowPass)

    a = []
    d = []

    for i in range(0, len(signal), 2):
        a.append(convolve(signal[i:i+L],lowPass,'valid')[0])
        d.append(convolve(signal[i:i+L],highPass,'valid')[0])

    return a,d


def idwt(coeffs, signal, wavelet):
    # Reconstruct original signal
    w = pywt.Wavelet(wavelet)
    lowPass = w.dec_lo
    highPass = np.flip(w.dec_hi)

    # Interleaf coefficients
    print(f'Approx: {len(coeffs[0])}')
    print(f'Detail: {len(coeffs[1])}')
    interleaved = np.insert(coeffs[1], np.arange(len(coeffs[0])), coeffs[0])

    # Pad Interleaved
    L = len(highPass)
    print(L)
    interleaved = np.append(interleaved, [1]*L)

    # Pad Reconstruction
    reconstructed = np.zeros((len(coeffs[0])*2 + L))

    for i in range(0,len(coeffs[0])*2,2):
        for j in range(L):
            reconstructed[i] += interleaved[i+j]*lowPass[j]
            reconstructed[i+1] += interleaved[i+j]*highPass[j]

    # Remove Padding
    P = len(coeffs[1])*2
    reconstructed = reconstructed[:P]
    print(f'Reconstructed Length (unpadded) {len(reconstructed)}')

    return reconstructed


def partial(signal, wavelet, level=3):
    # Leaf coefficients for approximate branch
    coeffs = []

    for l in range(level):
        (a,d) = dwt(signal, wavelet)
        signal = a
        coeffs.append(d)

    coeffs.append(a)

    return coeffs


def partialReconstruct(coeffs, signal, wavelet):
    # Reconstruct multilevel
    coeffs = np.flip(coeffs)
    recon = coeffs[0]

    for i in range(1,len(coeffs)):
        level = [recon, coeffs[i]]
        recon = idwt(level, signal, wavelet)

    return recon


# Chirp test signal
sampleRate = 1000
signalLength = 10
t = np.linspace(0, signalLength, int(sampleRate*signalLength))
chirpSignal = chirp(t, f0=1, f1=1, t1=5, method='linear')
noise = np.random.standard_normal(int(sampleRate*signalLength)) * 0.1
signal = chirpSignal + noise

wavelet = 'db2'
level = 2

#----------

# coeffs = dwt(signal, wavelet)

# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(coeffs[0])
# ax2.plot(coeffs[1])
# plt.show()

# reconstructed = idwt(coeffs, signal, wavelet)

# plt.plot(signal)
# plt.plot(reconstructed)
# plt.show()

#----------

signal = pad(signal, level, mode='circular')
print(f'Signal Length (Padded): {len(signal)}')

coeffs = partial(signal, wavelet, level)

fig, ax = plt.subplots(len(coeffs))
for i, coeff in enumerate(coeffs):
    ax[i].plot(coeff)
plt.show()

reconstructed = partialReconstruct(coeffs, signal, wavelet)

print(len(signal))
print(len(reconstructed))

L = 150
reconstructed = reconstructed[0:-L]
reconstructed = np.append([1]*L,reconstructed)

plt.plot(signal)
plt.plot(reconstructed)
plt.show()