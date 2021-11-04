import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
import matplotlib.pyplot as plt


def decomposition(signal):
    # Plot decomposition packets
    fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6,6))

    A = signal

    for ii in range(5):
        (A,D) = pywt.dwt(A, 'dmey')
        axarr[ii, 0].plot(A, 'r')
        axarr[ii, 1].plot(D, 'g')
        axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
        #axarr[ii, 0].set_yticklabels([])

        if ii == 0:
            axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
            axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    plt.tight_layout()
    plt.show()


# Chirp (Test signal)
t = np.linspace(0, 10, 1000)
signal = chirp(t, f0=0.1, f1=2, t1=10, method='linear')
noise = np.random.rand(1000) * 0.2
signal += noise

coeffs = pywt.wavedec(signal, wavelet='dmey', level=3)
print(len(coeffs))
#(cA2, cD2, cD1) = coeffs
# (cA3, cD2, cD2, cD1) = coeffs

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
# ax1.plot(cD1, label='cD1')
# ax1.legend()
# ax2.plot(cA2, label='cA2')
# ax2.legend()
# ax3.plot(cD2, label='cD2')
# ax3.legend()
# ax4.plot(cD3, label='cD3')
# ax4.legend()
# plt.show()

for i,coeff in enumerate(coeffs):
    if i != 0: # First index is the Approximate node, careful!
        thres = np.mean(abs(coeff))
        print(thres)
        coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft', substitute=0) # 0.2 works well
    

denoised = pywt.waverec(coeffs, wavelet='dmey')

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Original/Denoised Signal')
ax1.plot(signal)
ax2.plot(denoised)
plt.show()

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Original/Denoised Spectrogram')
ax1.specgram(signal, Fs=100)
ax2.specgram(denoised, Fs=100)
plt.show()

decomposition(signal)

