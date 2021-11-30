# manually constructed discrete wavelet transform based mostly on:
#https://mil.ufl.edu/nechyba/www/eel6562/course_materials/t5.wavelets/intro_dwt.pdf

import numpy as np
import pywt
from math import sqrt, floor, ceil
from scipy.signal import chirp
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy
import scipy.io.wavfile as wave


def pad(signal, wavelet, mode):
    #pad signal with additional values so that it is divisible by 2
    #add values to signal so that signal length is 2^n

    n = ceil(np.log10(len(signal))/np.log10(2))

    padded = dcopy(list(signal))

    if mode in ['per','periodic','peroidization']:
        for i in range(2**n-len(signal)):
            padded.append(signal[i]) #padding signal as if it is periodic

    elif mode in ['zero','zeros']:
        for i in range(2**n-len(signal)):
            padded.append(signal[i]) #padding signal as if it is zero outside time range

    else: 
        print(f"Mode '{mode}' not found")
        return None

    if 2**n-len(signal):
        print(f'Padded signal of length {len(signal)} with {2**n-len(signal)} values')

    return padded


def unpad(padded, signal, wavelet, mode):
    #remove extra values from padded signal

    print(f'Removed {len(padded)-len(signal)} padding values from reconstructed signal')
    padded = padded[:len(signal)]

    return padded


def pad2(signal,wavelet,mode):
    ###### not currently working properly
    #padding also affects the discrete and approximate coefficients if the filter is longer than 2
    #(see https://mil.ufl.edu/nechyba/www/eel6562/course_materials/t5.wavelets/intro_dwt.pdf for visualization)

    lowPass, highPass = getFilters(wavelet)

    filterLen = len(highPass) # high and low pass filters have same length
    signalLen = len(signal)

    padded = signal.copy()

    ###############     trying to implement pywt method of padding...


    # if not len(padded) == 2*floor((signalLen + filterLen - 1)/2): #padding signal - doesn't work the same as pywt...

    #     if mode == 'zero': # padding is added as if signal is zero outside time span
    #         while not len(padded) == 2*floor((signalLen + filterLen - 1)/2):
    #             if ((2*floor((signalLen + filterLen - 1)/2) - len(padded)) % 2) == 0: #alternate between padding beginning and end of signal until desired length
    #                 padded.append(0)
    #             else: padded.insert(0, 0)

    #     elif mode == 'per': # padding is added as if signal is periodic
    #         padding = 2*floor((signalLen + filterLen - 1)/2)-len(padded)
    #         for i in range(padding):
    #             padded.append(padded[i])

    #     else: 
    #         print(f"Mode '{mode}' not found")
    #         return None

    #     print(f'padded signal from a length of {signalLen} to a length of {len(padded)}')
    #     # print(f'padded signal: {padded}')
    return padded


def unpad2(signal, reconstructed, mode): 
    #### not currently working
    # remove values in reconstructed signal which emerge from padding

    while not len(reconstructed) == len(signal):
        if ((len(reconstructed)-len(signal)) % 2) == 0:
            reconstructed = reconstructed[:len(reconstructed)]
        else:
            reconstructed = reconstructed[1:]

    return reconstructed


def getFilters(wavelet, inverse=False):
    #stores high and low pass filter coefficients for decontruction and reconstruction for a variety of mother wavlets

    if wavelet == 'haar' or wavelet == 'db1': #haar wavelet / db1 wavelet (same coefficients)
        lowPass = [1/sqrt(2), 1/sqrt(2)] #forward low pass wavelet coefficients
        if inverse == True:
            lowPass = list(reversed(lowPass)) #inverse low pass wavelet coefficients

    elif wavelet == 'db2':
        lowPass = [(1-sqrt(3))/(4*sqrt(2)), (3-sqrt(3))/(4*sqrt(2)), (3+sqrt(3))/(4*sqrt(2)), (1+sqrt(3))/(4*sqrt(2))]
        if inverse == True:
            lowPass = list(reversed(lowPass)) #inverse low pass wavelet coefficients

    elif wavelet == 'halves': #wavelet used in example slide 4/98: https://www.math.aau.dk/digitalAssets/120/120646_r-2003-24.pdf
        lowPass = [1/2, 1/2]
        if inverse == True:
            lowPass = [1,1]
        
    else: 
        print(f'Filter {wavelet} not found.')
        return None

    highPass = []
    for i in range(len(lowPass)): #highpass filter can be constructed from lowpass filter according to: https://mil.ufl.edu/nechyba/www/eel6562/course_materials/t5.wavelets/intro_dwt.pdf
            j = len(lowPass)-1-i
            if inverse==False:
                if (i % 2):
                    highPass.append(-lowPass[j])
                else:
                    highPass.append(lowPass[j])

            elif inverse==True: #highpass filter is constructed from lowpass for inverse transform:
                if (i % 2):
                    highPass.append(-lowPass[j])
                else:
                    highPass.append(lowPass[j])

    return lowPass, highPass


def decompose(padded, wavelet, mode):
    #decompose signal into approximate and detail coefficients (1 level)

    lowPass, highPass = getFilters(wavelet)

    filterLen = len(highPass) # high and low pass filters have same length

    signalLen = len(padded)

    a = np.zeros(int(len(padded)/2)) #approximate coefficients
    d = np.zeros(len(a)) #detail coefficients

    # print(f'n = {np.log10(signalLen)/np.log10(2)} where signal length = 2^n\n')

    for i in range(0,signalLen,2): # step size of 2 results in a and d having 1/2 the length of signal
        for j in range(filterLen):
            a[int(i/2)] += padded[i+j]*lowPass[j] #lowpass reconstruction
            d[int(i/2)] += padded[i+j]*highPass[j] #highpass reconstruction
    
    return (a, d)


def reconstruct(decomposed, signal,wavelet,mode): 
    #reconstruct higher level from approx and detail coefficients of lower level
    
    lowPass, highPass = getFilters(wavelet, inverse=True)
    filterLen = len(highPass) # high and low pass filters have same length

    interleaved = np.empty(len(decomposed[0])*2) #first interleaf the approximate and detail coefficients
    interleaved[0::2] = decomposed[0] #approximate coeffs
    interleaved[1::2] = decomposed[1] #detail coeffs

    reconstructed = np.zeros(len(decomposed[0])*2)

    for i in range(0,len(decomposed[0])*2,2):
        # print(f'I {i}')
        for j in range(filterLen):
            # print(f'j {j}')
            reconstructed[i] += interleaved[i+j]*lowPass[j]
            reconstructed[i+1] += interleaved[i+j]*highPass[j]

    return reconstructed


def dwt(signal, level, wavelet, mode, plot=False):
    #multi-level decomposition
    coeffs = []

    approx = dcopy(signal)
    approx = pad(approx, wavelet, mode) #pad signal for decomposition

    for l in range(level):
        print(f'decompose level: {l}')
        [approx, detail] = decompose(approx, wavelet, mode) #get coeffs of single level decomposition
        coeffs.append(detail)
    
    coeffs.append(approx)

    # print(f'    Decomposition coefficients: {coeffs}\n')
    
    #plot signal decompositions
    if plot:
        fig, ax = plt.subplots(len(coeffs)+1)
        fig.suptitle('Signal decompositions')
        ax[0].plot(signal, color='black')
        ax[0].set_title('original signal')
        for i, coeff in enumerate(coeffs):
            ax[i+1].plot(coeff)
            if i == len(coeffs)-1:
                ax[i+1].set_title(f'level {i}, approx.')
            else: 
                ax[i+1].set_title(f'level {i+1}, detail')
        plt.show()

    return coeffs


def idwt(coeffs, signal, wavelet, mode, plot=False):
    #multi-level reconstruction 

    for i in range(len(coeffs)-1):
        approx = coeffs[len(coeffs)-1]
        detail = coeffs[len(coeffs)-2]

        # print(f'reconstruct level: {i}')

        if (len(coeffs) - 3) >= 0: #for determining the appropriate 'higher level' signal to base reconstruction on
            coeffs[len(coeffs)-2] = reconstruct([approx,detail],coeffs[len(coeffs)-3],wavelet,mode)
        else:
            coeffs[len(coeffs)-2] = reconstruct([approx,detail],signal,wavelet,mode)
        coeffs = coeffs[:len(coeffs)-1] #remove last value of coefficient array
    
    reconstructed = unpad(coeffs[0],signal,wavelet,mode)

    if plot:
        #plot signal reconstruction
        fig, ax = plt.subplots(2)
        fig.suptitle('original / reconstructed signal')
        ax[0].plot(signal, color='black')
        ax[0].set_title('original signal')
        ax[1].plot(reconstructed)
        ax[1].set_title('reconstructed signal')
        plt.show()

    return reconstructed


def multiRes(coeffs,signal,wavelet,mode):
    #plot multiresolution representation
    fig, ax = plt.subplots(len(coeffs)+1)
    fig.suptitle('Multiresolution')
    ax[0].plot(signal, color='black')
    ax[0].set_title('original signal')

    for i in range(len(coeffs)):
        zeroed = dcopy(coeffs)
        for j in range(len(zeroed)):
            if not j == i:
                zeroed[j] *= 0
        reconstructed = idwt(zeroed,signal,wavelet,mode)
        
        ax[i+1].plot(reconstructed)
        if i == len(coeffs)-1:
            ax[i+1].set_title(f'level {i}, approx.')
        else: 
            ax[i+1].set_title(f'level {i+1}, detail')

    plt.show()


signal = [56, 40, 8, 24, 48, 48, 40, 16]
# signal = [59,43,11,27,45,45,37,13]

#chirp test signal
sampleRate = 1000
signalLength = 10 #seconds
t = np.linspace(0, signalLength, int(sampleRate*signalLength))
chirpSignal = chirp(t, f0=1, f1=1, t1=5, method='linear')
noise = np.random.standard_normal(int(sampleRate*signalLength)) * 0.1
signal = chirpSignal + noise

#possum noise test
# sampleRate, signal = wave.read('recordings/original/PossumNoisy.wav')


wavelet = 'dmey'
mode = 'per'

level = 2

lowPass, highPass = getFilters(wavelet)
    
print(f'Wavelet: {wavelet} \n    Lowpass coefficients: {lowPass}\n    Highpass coefficients: {highPass} \n')

coeffs = dwt(signal, level, wavelet, mode, plot=True)

multiRes(coeffs,signal,wavelet,mode) #plot multiresolution plot of coeffs

reconstructed = idwt(coeffs, signal, wavelet, mode, plot=True) #reconstruct original signal from coeffs


# #check method against pywt's discrete wavelet transform:

w = pywt.Wavelet(wavelet)
print(f'pywt wavelet: {wavelet} \n    Lowpass coefficients: {w.dec_lo}\n    Highpass coefficients: {w.dec_hi} \n')
decomposed = pywt.dwt(signal,wavelet,mode) 
# print(f'    Approximate coeffs: {decomposed[0]}\n    Detail coeffs: {decomposed[1]}\n')
reconstructed = pywt.idwt(decomposed[0], decomposed[1], wavelet, mode)

#plot pywt signal and reconstructed signal
fig, ([ax1, ax2]) = plt.subplots(2)
fig.suptitle('PYWT Original / Reconstructed signal')
ax1.plot(signal,color='black')
# ax1.plot(signal-noise,color='red')
ax2.plot(reconstructed)

plt.show()
