import scipy.io.wavfile as wave
import numpy as np
import pywt
import matplotlib.pyplot as plt
import math

# Finding highest correlated mother wavelet (only partial tree)
#dmey performs well

def denoise(signal,wavelet, level): #partial tree

    coeffs = pywt.wavedec(signal,wavelet,mode='symmetric')

    thres = np.std(coeffs[len(coeffs)-1])

    for i,coeff in enumerate(coeffs):
        if i != 0: # First index is the Approximate node, careful!
            thres = 10 #override threshold
            thres = thres * 4.5 # AviaNZ suggests the std of the lowest packet x4.5
            coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # 0.2 works well for chirp

    denoised = pywt.waverec(coeffs,wavelet) #reconstruct signal from partial tree

    if not len(denoised)==len(signal): #eliminate padding term in denoised signal, due to rounding up when bandwidth is halved
        denoised = denoised[:len(denoised)-1]

    return denoised

sampleRate, signal = wave.read('recordings/downsampled/cat16k.wav') # possum.wav works well Haar, 5, partial, thres=96*4.5
form = signal.dtype

# print((2*np.log2(len(signal)/sampleRate))**0.5)

# print(pywt.wavelist(kind='discrete'))

correlations = np.array(['correlation coefficient','mother wavelet'])

for wavelet in pywt.wavelist(kind='discrete'):
    # wavelet = 'dmey'
    level = 6

    denoised = denoise(signal, wavelet=wavelet, level=level)

    correlation = [np.corrcoef(signal, denoised)[1][0],wavelet]
    correlations = np.vstack([correlations,correlation]) #calculate correlation of signal and denoised signal using aviaNZ method
    # correlation = sum((signal-np.mean(signal))*(denoised-np.mean(denoised)))/(sum((signal-np.mean(signal))**2)*sum((denoised-np.mean(denoised))**2))**0.5

print(correlations[np.where(correlations==max(correlations[1:,0]))[0]]) #mother wavelet with maximum correlation
print(correlations[np.where(correlations=='dmey')[0]]) #dmey wavelet for comparison

#display for best mother wavelet
wavelet = correlations[np.where(correlations==max(correlations[1:,0]))[0]][0][1]
denoised = denoise(signal, wavelet=wavelet, level=level)

# plt.figure()
# plt.title('Original/Denoised signal')
# plt.plot(signal)
# plt.plot(denoised)

# fig, (ax1, ax2) = plt.subplots(2)
# fig.suptitle('Original/Denoised Spectrogram')
# ax1.specgram(signal, Fs=sampleRate)
# denoised = np.asarray(denoised, dtype=form) # Downsample
# ax2.specgram(denoised, Fs=sampleRate)
# plt.show()