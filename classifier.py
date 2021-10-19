import scipy.io.wavfile as wave
from scipy import signal 
import os
import matplotlib.pyplot as plt
import pywt
import numpy as np
import difflib

import skimage.measure    
#entropy = skimage.measure.shannon_entropy(img)


def entropy(data):
    E=data**2/len(data)
    P=E/sum(E)
    S=-sum(P*np.log2(P))
    return S 

# Random signal
noise = np.random.normal(0, 1, 1000)

# Signal
sampleRate, signal = wave.read('./recordings/miaow_16k.wav')


# Spectrogram 

# plt.specgram(signal, Fs=sampleRate)
# plt.savefig('spec.png')

with open('spec.png', 'r') as spec:
    entropy = skimage.measure.shannon_entropy(spec)
    print(entropy)

# plt.show()
data = signal

#Wavelet Packet Decomposition
fig, axarr = plt.subplots(nrows=5, ncols=4, figsize=(6,6))
for ii in range(5):
    (data, coeff_d) = pywt.dwt(data, 'sym5')
    axarr[ii, 0].plot(data, 'r')
    axarr[ii, 1].specgram(data,Fs=sampleRate)
    axarr[ii, 2].plot(coeff_d, 'g')
    axarr[ii, 3].specgram(coeff_d,Fs=sampleRate)
    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
    axarr[ii, 0].set_yticklabels([])
    if ii == 0:
        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    axarr[ii, 1].set_yticklabels([])
plt.tight_layout()
plt.show()

# using wavedec
coeffs = pywt.wavedec(signal, 'sym5', level=5)

a = [1,2,3,4,5]
b = [1,0,3,4,5]

sm = difflib.SequenceMatcher(None,coeffs[0],coeffs[1])
print(sm.ratio())

# Remove coeffs (Use Shannon Entropy)
for i,coeff in enumerate(coeffs):
    en = entropy(coeff)
    print(en)
    if en > 12:
        coeffs[i] = np.zeros_like(coeffs[i])

print(coeffs)

# Reconstruct signal
reconstructed_signal = pywt.waverec(coeffs, 'sym5')

plt.figure(2)
plt.plot(signal)
plt.plot(reconstructed_signal)
plt.show()



# # If spectral range exceeds thresgold, recommend
# frequency, time, spectrogram = signal.spectrogram(data, sampleRate)

# # print(frequency)
# print(spectrogram)

# t = []

# for instance in spectrogram:
#     flag = 0
#     for sample in instance:
#         if sample > 60:
#             flag = 1
#             break
#         else:
#             flag = 0

#     t.append(flag)
       
# print(t)

# plt.specgram(data,Fs=sampleRate)
# plt.xlabel('Time [sec]')
# plt.ylabel('Frequency [Hz]')
# plt.show()