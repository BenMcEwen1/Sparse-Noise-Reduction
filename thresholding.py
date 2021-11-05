import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
import matplotlib.pyplot as plt


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

def partialTree(signal, plot=False):
    # Returns leaf coeffs of approximate branch
    coeffs = pywt.wavedec(signal, wavelet='dmey',mode='symmetric', level=3)

    if plot:
        (cA3, cD3, cD2, cD1) = coeffs
        fig, ax = plt.subplots(4)
        ax[0].plot(cD1, label='cD1')
        ax[1].plot(cD2, label='cD2')
        ax[2].plot(cD3, label='cD3')
        ax[3].plot(cA3, label='cA3')
        plt.show()

    return coeffs

def fullTree(signal, wavelet='dmey', level=3):
    # Return leaf coeffs of full tree
    A = signal
    D = None

    coeffs = []
    for i in range(level):
        if i == 0: # First node so only one packet 
            (A,D) = pywt.dwt(signal, wavelet)
            coeffs.append(A)
            coeffs.append(A)
        else:
            L = []
            for coeff in coeffs:
                
                (A,D) = pywt.dwt(coeff, wavelet)
                L.append(A) 
                L.append(D) 
            
            coeffs = L

    print((len(L)))

    # (cA3, cD3, cD2, cD1) = L
    # fig, ax = plt.subplots(4)
    # ax[0].plot(cD1, label='cD1')
    # ax[1].plot(cD2, label='cD2')
    # ax[2].plot(cD3, label='cD3')
    # ax[3].plot(cA3, label='cA3')
    # plt.show()






# Chirp (Test signal)
sampleRate = 1000
t = np.linspace(0, 10, sampleRate)
signal = chirp(t, f0=0.1, f1=2, t1=10, method='linear')
noise = np.random.standard_normal(sampleRate) * 0.1
signal += noise

# Noisy possum Test
# sampleRate, signal = wave.read('recordings/PossumNoisy.wav')
# form = signal.dtype
# level = 6
# print(sampleRate)

fullTree(signal)

# # Calculate coefficients
# #coeffs = partialTree(signal, plot=False)

# coeffs = pywt.wavedec(signal, wavelet='dmey',mode='symmetric', level=level)

# #down_coeffs = pywt.downcoef(part='d', data=signal, wavelet='dmey', mode='symmetric', level=5) # Allows you to pick out individual packets, tree is the same as wavedec though

# # Apply thresholding to detailed coeffs
# for i,coeff in enumerate(coeffs):
#     if i != 0: # First index is the Approximate node, careful!
#         thres = 2*np.std(coeff)
#         print(thres)
#         coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # 0.2 works well for chirp
#         # coeffs[i] = np.zeros(len(coeff))


# denoised = pywt.waverec(coeffs, wavelet='dmey')

# plt.figure()
# plt.title('Original/Denoised signal')
# plt.plot(signal)
# plt.plot(denoised)
# plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.suptitle('Original/Denoised Spectrogram')
# ax1.specgram(signal, Fs=sampleRate)
# ax2.specgram(denoised, Fs=sampleRate)
# denoised = np.asarray(denoised, dtype=form) # Downsample
# ax3.specgram(denoised, Fs=sampleRate)
# plt.show()

# decomposition(signal, level)

# # Save denoised signal
# wave.write('denoised/denoised.wav', sampleRate, denoised)