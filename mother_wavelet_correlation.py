import scipy.io.wavfile as wave
import numpy as np
import pywt

def denoise(signal,wavelet, level): #partial tree

    coeffs = pywt.wavedec(signal,wavelet,mode='symmetric')
    thres = np.std(coeffs[len(coeffs)-1])

    for i,coeff in enumerate(coeffs):
        if i != 0: # First index is the Approximate node, careful!
            thres = 10 # Override threshold
            thres = thres * 4.5 # AviaNZ suggests the std of the lowest packet x4.5
            coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # 0.2 works well for chirp

    denoised = pywt.waverec(coeffs,wavelet) # Reconstruct signal from partial tree

    if not len(denoised)==len(signal): # Eliminate padding term in denoised signal, due to rounding up when bandwidth is halved
        denoised = denoised[:len(denoised)-1]

    return denoised

sampleRate, signal = wave.read('recordings/downsampled/cat16k.wav') # possum.wav works well Haar, 5, partial, thres=96*4.5
form = signal.dtype


correlations = np.array(['correlation coefficient','mother wavelet'])

for wavelet in pywt.wavelist(kind='discrete'):
    level = 6

    denoised = denoise(signal, wavelet=wavelet, level=level)

    correlation = [np.corrcoef(signal, denoised)[1][0],wavelet]
    correlations = np.vstack([correlations,correlation]) # Calculate correlation of signal and denoised signal using aviaNZ method

print(correlations[np.where(correlations==max(correlations[1:,0]))[0]]) # Mother wavelet with maximum correlation
print(correlations[np.where(correlations=='dmey')[0]]) # Dmey wavelet for comparison

# Display for best mother wavelet
wavelet = correlations[np.where(correlations==max(correlations[1:,0]))[0]][0][1]
denoised = denoise(signal, wavelet=wavelet, level=level)
