import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
from scipy.stats import entropy, kstest, uniform
import matplotlib.pyplot as plt
import time
from math import ceil

# def entropy(data):
#     # Calculate the entropy given packet coefficients
#     # print(data)
#     norm = np.linalg.norm(data)
#     e = data[np.nonzero(norm)]**2 * np.log2(data[np.nonzero(norm)]**2)
#     return -np.sum(e)

def KS(data):
    # Kolmogororov-Smirnov test of uniformity
    Uniformity = kstest(data, uniform(loc=0.0, scale=len(data)).cdf)
    return Uniformity.statistic


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


def decomposeFull(signal, wavelet='dmey', levels=5, plot=False):
    # Return leaf coeffs of full tree
    coeffs = [signal]

    for i in range(levels):
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


def thresholdFull(signal, thres, wavelet='dmey', levels=5):
    e = entropy(abs(signal))
    stop = False

    for l in range(1,levels+1): # For final leaf nodes set the start range = levels
        coeffs = decomposeFull(signal, wavelet=wavelet, levels=l, plot=False)

        currentE = []

        for coeff in coeffs:
            # coeff = np.divide(coeff, sum(coeff))
            currentE.append(entropy(abs(coeff)))

        if max(currentE) > e:
            print(f"Stopping at level: {l}")
            stop = True
        else:
            e = max(currentE) 
        
        print(f"Number of leaves: {len(coeffs)}")
        print(f"coeffs per leaf: {len(coeffs[0])}")

        # # aviaNZ thresholding method
        # stdevs = []
        # for coeff in coeffs[int(len(coeffs)/2):]:
        #     stdevs.append(np.std(coeff))
        # thres = 4.5*np.mean(stdevs)
        # for i,coeff in enumerate(coeffs):
        #     # coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # 0.2 works well for chirp
        #     for j,val in enumerate(coeff):
        #         if abs(val) < thres:
        #             #coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # 0.2 works well for chirp
        #             coeff[j] = coeff[j]*0 #hard thresholding
        
        #box search method
        if l == level: #only thresholding at the lowest level
            # check maximum entropy at frequency (leaf number) 
            thres = 0
            localboxCos = None
            maxEntropy = 0

            # search through coefficient array with boxes to find highest entropy region, treat this as noise
            # divide the coefficient array dimensions by 10 for the box size
            boxHeight = len(coeffs)/10
            boxWidth = len(coeffs[0])/10
            for i in range(ceil(boxHeight/2), len(coeffs)-ceil(boxHeight/2), ceil(boxHeight/4)):
                for j in range(0, len(coeffs[i])-ceil(boxWidth), ceil(boxWidth/2)): #iterate over array with stepsize of 1/2 the box width/height

                    localBox = np.array(coeffs)[i-ceil(boxHeight/2):i+ceil(boxHeight/2), j:j+ceil(boxWidth)]

                    if entropy(abs(localBox.flatten())) > maxEntropy: #store coefficients from box with maximum entropy
                        maxEntropy = entropy(abs(localBox.flatten()))
                        localboxCos = [i-ceil(len(coeffs)/20),i+ceil(len(coeffs)/20),j,j+ceil(len(coeffs[i])/10)]
                        thres = 4.5*np.std(localBox) #thres is 4.5 x standard deviation of 'noise' to cover 99.99% of noise

            # Apply thresholding to all coeffs
            coeffs = np.array(coeffs)
            if not localboxCos == None:
                for i,coeff in enumerate(coeffs):
                    for j,val in enumerate(coeff):
                        if abs(val) < thres:
                            coeff[j] = coeff[j]*0.2 #soft thres-holding

                # # display the region being treated as noise by setting this to low
                # coeffs[localboxCos[0]:localboxCos[1], localboxCos[2]:localboxCos[3]] = coeffs[localboxCos[0]:localboxCos[1], localboxCos[2]:localboxCos[3]]*0

        # # Apply thresholding to detailed coeffs
        # for i,coeff in enumerate(coeffs):
        #     # thres = 0.2*np.std(coeff)
        #     thres = thres * 0.6
        #     # thres = universalThresholding(coeff) # Use garrote
        #     # print(thres)
        #     # thres = thres * 0.002
        #     coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # soft or garote works well

        # Reconstruct each level
        signal = reconstructFull(coeffs, wavelet=wavelet, plot=False) # Full tree reconstruction

        if stop:
            break

    return signal


def thresholdPartial(signal, thres, wavelet='dmey', level=5):
    coeffs = partialTree(signal, levels=level, plot=True)

    for i,coeff in enumerate(coeffs):
        if i != 0: # First index is the Approximate node, careful!
            thres = thres * 0.6 # AviaNZ suggests the std of the lowest packet x4.5
            # thres = universalThresholding(coeff)
            coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # 0.2 works well for chirp

    signal = pywt.waverec(coeffs, wavelet='dmey')

    return signal


def findThres(signal, maxLevel):
    coeffs = decomposeFull(signal, levels=maxLevel)
    return np.std(coeffs[0])


def uniformity(signal, maxLevel):
    coeffs = decomposeFull(signal, levels=maxLevel)
    s = []
    for coeff in coeffs:
        if KS(coeff) > 0.90:
            s.append(np.std(coeffs))

    return np.average(s)

def universalThresholding(coeffs):
    v = np.median(abs(coeffs))/0.6745
    N = len(coeffs)

    return v * np.sqrt(2*np.log(N))


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
sampleRate, signal = wave.read('recordings/PossumNoisy.wav') # possum.wav works well Haar or dmey, 5, partial, thres=96*4.5
form = signal.dtype
wavelet = 'dmey'

level = pywt.dwt_max_level(len(signal), wavelet) # Calculate the maximum level
# level = 5

# Normalisation
# signal = (signal - np.mean(signal)) / np.std(signal) # Normalisation
# signal = np.divide(signal, sum(signal)) # Generate random variable that add to 1.0

# print(pywt.wavelist(kind='discrete'))
print(f"Max level: {level}")

thres = findThres(signal, level)
# thres = universalThresholding(signal)
print(thres)

# decomposition(signal, level)


denoised = thresholdFull(signal, thres, wavelet=wavelet, levels=level)
# denoised = thresholdPartial(signal, thres, wavelet=wavelet, level=level)


plt.figure()
plt.title('Original/Denoised signal')
plt.plot(signal)
plt.plot(denoised)
plt.show()

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Original/Denoised Spectrogram')
ax1.specgram(signal, Fs=sampleRate)
# denoised = np.asarray(denoised, dtype=form) # Downsample
ax2.specgram(denoised, Fs=sampleRate)

plt.show()

# Save denoised signal
wave.write('denoised/denoised.wav', sampleRate, denoised)