import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
from scipy.stats import entropy, kstest, uniform
import matplotlib.pyplot as plt
import time
from math import sqrt

import colorednoise as cn

# Following the full-frequency denoising method

# steps involved:
#   1) find noise type
#   2) decompose into approx and detail
#   3) threshold approx and detail with their respective functions (Low and High frequency thresholding)
#   4) check if approx still has noise, if it does:
#       repeat from step 2
#   5) reconstruct signal from denoised coefficients


def autoCorrelation(a, cutoff): # get (cutoff) number of autocorrelation coeffs of a signal
    correlation = np.correlate(a,a,mode='full')[len(a)-1:]/np.var(a)/len(a)
    correlation = correlation[:cutoff]
    return correlation


def getNoiseCorrs(samples, correlationLength, plot=False): #get list of auto-correlation coefficients for each noise type 

    colors = ['white','pink','red', 'blue', 'violet']
    noiseACs = [] #noise auto-correlations

    for color in [0, 1, 2, -1, -2]: # power coefficients of each color of noise
        noise = cn.powerlaw_psd_gaussian(color, samples) #Generate coloured noise
        correlation = autoCorrelation(noise, correlationLength)
        noiseACs.append(correlation)
    
    if plot:
        plt.figure()
        for i in noiseACs:
            plt.plot(i)
        plt.legend(colors)
        plt.show()

    return noiseACs, colors


def getNoiseType(signal):

    samples = 2**13
    correlationLength = 40 #number of auto-correlation coefficients, greater = more accurate but slower
    critical = 0.4

    noiseACs, noiseTypes = getNoiseCorrs(samples, correlationLength, plot=False) #noise auto-correlation coefficients
    sigAC = autoCorrelation(signal[:np.min([samples,len(signal)])], correlationLength) #signal auto-correlation coefficient

    SDS = np.empty(len(noiseACs))
    for i, noiseAC in enumerate(noiseACs):
        SDS[i] = np.sum( (noiseAC - sigAC)**2 )
    noiseType = np.array(noiseTypes)[np.where(SDS==np.min(SDS))][0]

    #check if signal is noisy using wavelet 'de-correlation'
    noiseAC = np.array(noiseACs)[np.where(SDS==np.min(SDS))][0]
    F = np.sum(np.square(noiseAC))/np.sum(np.square(sigAC)) - 1
    if abs(F) > critical:
        noisy = True
    else: noisy = False

    return noiseType, noisy


def threshold(coeffs, noiseType, level):
    approx = coeffs[0]
    detail = coeffs[1]
    
    #threshold values and function given in Section 4.4 https://www.sciencedirect.com/science/article/pii/S0925231216307688
    def thres(sigma, level, m, a, b, c):
        if (a*level**2 + b*level + c)/m > 0:
            thres = sigma*sqrt((a*level**2 + b*level + c)/m)
        else: thres = 0
        return thres
    
    CVA = np.std(approx)/np.mean(approx)  
    CVD = np.std(detail)/np.mean(detail)

    sigma = 1# noise intensity estimation, should be related to coefficient of variation

    if noiseType == 'white':
        HF = [35.84, -444.5, 1348]
        LF = [46.72, -504.7, 1413]
    elif noiseType == 'blue':
        HF = [60.99, -701.4, 1858]
        LF = [23.55, -270.2, 713.9]
    elif noiseType == 'violet':
        HF = [71.93, -807.8, 2059]
        LF = [16.95, -174.3, 425.1]
    elif noiseType == 'pink':
        HF = [-0.05978, -8462, 191.1]
        LF = [6.081, -244.2, 2096]
    elif noiseType == 'red':
        HF = [0.1522, 9.285, -12.87]
        LF = [-0.961, -9.907, 2063]

    thresH = thres(sigma, level, len(approx), HF[0], HF[1], HF[2])
    thresL = thres(sigma, level, len(approx), LF[0], LF[1], LF[2])

    print(f'level {level}, timesamples: {len(approx)} thresA-thresD = {thresL},{thresH}')

    for i, coeff in enumerate(cA):
        if abs(coeff) < thresL:
            approx[i] *= 0
    for i, coeff in enumerate(cD):
        if abs(coeff) < thresH:
            detail[i] *= 0

    return [approx, detail]


sampleRate, signal = wave.read('recordings/original/cat.wav') # possum.wav works well Haar or dmey, 5, partial, thres=96*4.5

noiseType, noisy = getNoiseType(signal)

print(f'noise matches {noiseType} most closely')

wavelet = 'dmey'
levels = 8

reconstructCoeffs = [signal]
cA = signal

for level in range(1,levels+1):

    [cA, cD] = pywt.dwt(cA, wavelet)

    if noisy:
        [cA, cD] = threshold([cA, cD], noiseType, level)

    noiseType, noisy = getNoiseType(cA)

    print(f'noise matches {noiseType} most closely')

    reconstructCoeffs[0] = cD
    reconstructCoeffs.insert(0, cA)

reconstructed = pywt.waverec(reconstructCoeffs, wavelet)

plt.figure()
plt.title('Original/Denoised signal')
plt.plot(signal,color='black')
plt.plot(reconstructed)

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Original/Denoised Spectrogram')
ax1.specgram(signal, Fs=sampleRate)
form = signal.dtype
denoised = np.asarray(reconstructed, dtype=form) # Downsample
ax2.specgram(denoised, Fs=sampleRate)


# Save denoised signal
wave.write('denoised/denoised.wav', sampleRate*int(len(denoised)/len(signal)), denoised)

plt.show()
