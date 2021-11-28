import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
from scipy.stats import entropy, kstest, uniform
import matplotlib.pyplot as plt
import time

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

    noiseACs, noiseTypes = getNoiseCorrs(samples, correlationLength, plot=False) #noise auto-correlation coefficients
    sigAC = autoCorrelation(signal[:np.min([samples,len(signal)])], correlationLength) #signal auto-correlation coefficient

    SDS = np.empty(len(noiseACs))
    for i, noiseAC in enumerate(noiseACs):
        SDS[i] = np.sum( (noiseAC - sigAC)**2 )
    noiseType = np.array(noiseTypes)[np.where(SDS==np.min(SDS))][0]

    return noiseType


sampleRate, signal = wave.read('recordings/original/cat.wav') # possum.wav works well Haar or dmey, 5, partial, thres=96*4.5

noiseType = getNoiseType(signal)

print(f'noise matches {noiseType} most closely')




