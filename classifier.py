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
sampleRate, signal = wave.read('./recordings/cat.wav')

# Generate Tree
wp = pywt.WaveletPacket(signal, wavelet='dmey', mode='symmetric', maxlevel=3)

(data, coeff_d) = pywt.dwt(signal, 'dmey')

plt.figure(1)
plt.specgram(data,Fs=sampleRate)
plt.show()

for node in wp.get_level(3, 'freq'):
    print(node.path)
    if (node.path == 'aaa'):
        #print("delete")
        del wp[node.path]

        # before = node.data

        # print("Attentuate")
        # #wp[node.path].data = pywt.threshold(node.data, 10, mode='hard', substitute=0)
        # wp[node.path].data = node.data * 0.2

        # plt.plot(before)
        # plt.plot(node.data)
        # plt.show()


denoised = wp.reconstruct(update=True)

plt.figure(2)
plt.plot(signal)
plt.plot(denoised)
plt.show()

plt.figure(3)
plt.specgram(denoised,Fs=sampleRate)
plt.show()


# for i in range(wp.maxlevel):
#     print([node.path for node in wp.get_level(i+1, 'freq')])


# # Spectrogram 

# # plt.spdata = signal

# #Wavelet Packet Decomposition
# fig, axarr = plt.subplots(nrows=5, ncols=4, figsize=(6,6))
# for ii in range(5):
#     (data, coeff_d) = pywt.dwt(data, 'sym5')
#     axarr[ii, 0].plot(data, 'r')
#     axarr[ii, 1].specgram(data,Fs=sampleRate)
#     axarr[ii, 2].plot(coeff_d, 'g')
#     axarr[ii, 3].specgram(coeff_d,Fs=sampleRate)
#     axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
#     axarr[ii, 0].set_yticklabels([])
#     if ii == 0:
#         axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
#         axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
#     axarr[ii, 1].set_yticklabels([])
# plt.tight_layout()
# plt.show()ecgram(signal, Fs=sampleRate)
# # plt.savefig('spec.png')

# with open('spec.png', 'r') as spec:
#     entropy = skimage.measure.shannon_entropy(spec)
#     print(entropy)

# # plt.show()
# data = signal

# #Wavelet Packet Decomposition
# fig, axarr = plt.subplots(nrows=5, ncols=4, figsize=(6,6))
# for ii in range(5):
#     (data, coeff_d) = pywt.dwt(data, 'sym5')
#     axarr[ii, 0].plot(data, 'r')
#     axarr[ii, 1].specgram(data,Fs=sampleRate)
#     axarr[ii, 2].plot(coeff_d, 'g')
#     axarr[ii, 3].specgram(coeff_d,Fs=sampleRate)
#     axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
#     axarr[ii, 0].set_yticklabels([])
#     if ii == 0:
#         axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
#         axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
#     axarr[ii, 1].set_yticklabels([])
# plt.tight_layout()
# plt.show()

# # using wavedec
# coeffs = pywt.wavedec(signal, 'sym5', level=5)

# a = [1,2,3,4,5]
# b = [1,0,3,4,5]

# sm = difflib.SequenceMatcher(None,coeffs[0],coeffs[1])
# print(sm.ratio())

# # Remove coeffs (Use Shannon Entropy)
# for i,coeff in enumerate(coeffs):
#     #en = entropy(coeff)
#     #print(en)
#     if i == 5:
#         coeffs[i] = np.zeros_like(coeffs[i])

# print(coeffs)

# # Reconstruct signal
# reconstructed_signal = pywt.waverec(coeffs, 'sym5')

# plt.figure(2)
# plt.plot(signal)
# plt.plot(reconstructed_signal)
# plt.show()

a = wp['a'].data #First Node
d = wp['d'].data #Second Node
#The second floor
aa = wp['aa'].data 
ad = wp['ad'].data 
dd = wp['dd'].data 
da = wp['da'].data 
#Layer 3
aaa = wp['aaa'].data 
aad = wp['aad'].data 
ada = wp['add'].data 
add = wp['ada'].data 
daa = wp['dda'].data 
dad = wp['ddd'].data 
dda = wp['dad'].data 
ddd = wp['daa'].data

plt.figure(figsize=(15, 10))
 
plt.subplot(4,1,1)
plt.plot(signal)
#First floor
plt.subplot(4,2,3)
plt.plot(a)
plt.subplot(4,2,4)
plt.plot(d)
#The second floor
plt.subplot(4,4,9)
plt.plot(aa)
plt.subplot(4,4,10)
plt.plot(ad)
plt.subplot(4,4,11)
plt.plot(dd)
plt.subplot(4,4,12)
plt.plot(da)
#Layer 3
plt.subplot(4,8,25)
plt.plot(aaa)
plt.subplot(4,8,26)
plt.plot(aad)
plt.subplot(4,8,27)
plt.plot(add)
plt.subplot(4,8,28)
plt.plot(ada)
plt.subplot(4,8,29)
plt.plot(dda)
plt.subplot(4,8,30)
plt.plot(ddd)
plt.subplot(4,8,31)
plt.plot(dad)
plt.subplot(4,8,32)
plt.plot(daa)
plt.show()