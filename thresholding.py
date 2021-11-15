import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
from scipy.stats import entropy
import matplotlib.pyplot as plt
import time


# def entropy(data):
#     # Calculate the entropy given packet coefficients
#     # print(data)
#     norm = np.linalg.norm(data)
#     e = data[np.nonzero(norm)]**2 * np.log2(data[np.nonzero(norm)]**2)
#     return -np.sum(e)




def decomposition(signal, level):
    # Plot decomposition packets
    fig, axarr = plt.subplots(nrows=level, ncols=2, figsize=(6,6))

    A = signal

    for ii in range(level):
        (A,D) = pywt.dwt(A, 'dmey')
        axarr[ii, 0].plot(A, 'r')
        axarr[ii, 1].plot(D, 'g')
        axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)

        if ii == 0:
            axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
            axarr[ii, 1].set_title("Detail coefficients", fontsize=14)

    plt.tight_layout()
    plt.show()


def partialTree(signal, levels=5, plot=False):
    # Returns leaf coeffs of approximate branch
    coeffs = pywt.wavedec(signal, wavelet='dmey',mode='symmetric', level=levels)

    if plot:
        fig, ax = plt.subplots(len(coeffs))
        for i, coeff in enumerate(coeffs):
            ax[i].plot(coeff)
        plt.show()

    return coeffs


def decomposeFull(signal, wavelet='dmey', level=5, plot=False):
    # Return leaf coeffs of full tree
    coeffs = [signal]

    for i in range(level):
        temp = []
        for coeff in coeffs:
            (A,D) = pywt.dwt(coeff, wavelet)
            temp.append(A) 
            temp.append(D) 
        
        coeffs = temp

    if plot:
        fig, ax = plt.subplots(len(coeffs))
        for i, coeff in enumerate(coeffs):
            ax[i].plot(coeff)
        plt.show()

    return coeffs


def reconstructFull(coeffs, wavelet='dmey', plot=False):
    # Reconstruct full wavelet tree
    upper = []
    levels = int(np.log2(len(coeffs)))

    for l in range(levels):
        for i in range(1, len(coeffs), 2):
            U = pywt.idwt(coeffs[i-1], coeffs[i], wavelet)
            upper.append(U)
        coeffs = upper
        upper = []

    if plot:
        fig, ax = plt.subplots(len(coeffs))
        for i, coeff in enumerate(coeffs):
            ax[i].plot(coeff)
        plt.show()

    return coeffs[0]


def thresholdFull(signal, wavelet='dmey', level=5):
    e = entropy(abs(signal))
    stop = False
    print(f"Initial Entropy: {e}")

    for l in range(1,level+1):
        coeffs = decomposeFull(signal, wavelet=wavelet, level=l, plot=False)

        print(f"Level: {l}")
        currentE = []

        for coeff in coeffs:
            # coeff = np.divide(coeff, sum(coeff))
            currentE.append(entropy(abs(coeff)))

        if max(currentE) > e:
            print(f"Stopping at level: {l}")
            stop = True
        else:
            print(f"Current max {max(currentE)}, previous max {e}")
            e = max(currentE) 

        # Apply thresholding to detailed coeffs
        for i,coeff in enumerate(coeffs):
            # if i != 0:
            thres = 1*np.std(coeff)
            thres = 10
            coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # 0.2 works well for chirp

        # Reconstruct each level
        signal = reconstructFull(coeffs, wavelet=wavelet, plot=False) # Full tree reconstruction

        if stop:
            break

    return signal


def thresholdPartial(signal, wavelet='dmey', level=5):
    coeffs = partialTree(signal, levels=level, plot=False)

    for i,coeff in enumerate(coeffs):
        if i != 0: # First index is the Approximate node, careful!
            thres = 1*np.std(coeff)
            print(thres)
            thres = 96 # AviaNZ suggests the std of the lowest packet x4.5
            coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # 0.2 works well for chirp

    signal = pywt.waverec(coeffs, wavelet='dmey')

    return signal


# # Chirp (Test signal)
# sampleRate = 1000
# t = np.linspace(0, 10, sampleRate)
# signal = chirp(t, f0=0.1, f1=2, t1=10, method='linear')
# noise = np.random.standard_normal(sampleRate) * 0.1
# signal += noise
# wavelet = 'dmey'
# level = pywt.dwt_max_level(len(signal), wavelet) # Calculate the maximum level
# form = signal.dtype


# Noisy possum Test
sampleRate, signal = wave.read('recordings/cat.wav') # possum.wav works well Haar or dmey, 5, partial, thres=96*4.5
form = signal.dtype
wavelet = 'dmey'

level = pywt.dwt_max_level(len(signal), wavelet) # Calculate the maximum level
# level = 5

# Normalisation
# signal = (signal - np.mean(signal)) / np.std(signal) # Normalisation
# signal = np.divide(signal, sum(signal)) # Generate random variable that add to 1.0

# print(pywt.wavelist(kind='discrete'))

print(f"Max level: {level}")
denoised = thresholdFull(signal, wavelet=wavelet, level=level)
# denoised = thresholdPartial(signal, wavelet=wavelet, level=level)

plt.figure()
plt.title('Original/Denoised signal')
plt.plot(signal)
plt.plot(denoised)
plt.show()

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Original/Denoised Spectrogram')
ax1.specgram(signal, Fs=sampleRate)
denoised = np.asarray(denoised, dtype=form) # Downsample
ax2.specgram(denoised, Fs=sampleRate)

plt.show()

# Save denoised signal
wave.write('denoised/denoised.wav', sampleRate, denoised)