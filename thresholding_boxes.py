import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
from scipy.stats import entropy, kstest, uniform
import matplotlib.pyplot as plt
import time
from math import ceil
from matplotlib.patches import Polygon

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
        
        print(f"Level {l}")
        print(f"Number of leaves: {len(coeffs)}")
        print(f"Coeffs per leaf: {len(coeffs[0])}")

        if False:
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
            pass
        
        #box search method
        if l == level: #only thresholding at the lowest level
            # check maximum entropy at frequency (leaf number) 
            threses = []
            vertStrips = []

            # search through coefficient array with boxes to find highest entropy region, treat this as noise
            # divide the coefficient array dimensions by 10 for the box size
            [boxHeight,boxWidth] = [ceil(len(coeffs)/10)*2,ceil(len(coeffs[0])/20)*2]
            for j in range(0, len(coeffs[0])-int(boxWidth/2), int(boxWidth/2)): #iterate over array with stepsize of 1/2 the box width/height
                vertStrips.append([])
                threses.append([])
                maxEntropy = 0

                for i in range(int(boxHeight/2), len(coeffs)-int(boxHeight/4), int(boxHeight/2)):

                    localBox = np.array(coeffs)[i-int(boxHeight/2):i+int(boxHeight/2), j:j+boxWidth]
                    if entropy(abs(localBox.flatten())) > maxEntropy: #store coefficients from box with maximum entropy
                        maxEntropy = entropy(abs(localBox.flatten()))
                        localboxCos = [i-int(boxHeight/2),i+int(boxHeight/2),j,j+boxWidth]
                        threses[len(threses)-1] = 4.5*np.std(localBox) #thres is 4.5 x standard deviation of 'noise' to cover 99.99% of noise

                vertStrips[len(vertStrips)-1] = localboxCos
            
            #horizontal box section
            threses2 = []
            horStrips = []
            [boxHeight,boxWidth] = [ceil(len(coeffs)/40)*2,ceil(len(coeffs[0])/20)*2]
            
            for i in range(int(boxHeight/2), len(coeffs)-int(boxHeight/4), int(boxHeight/2)):
                horStrips.append([])
                threses2.append([])
                maxEntropy = 0

                for j in range(0, len(coeffs[0])-int(boxWidth/2), int(boxWidth/2)): #iterate over array with stepsize of 1/2 the box width/height

                    localBox = np.array(coeffs)[i-int(boxHeight/2):i+int(boxHeight/2), j:j+boxWidth]
                    if entropy(abs(localBox.flatten())) > maxEntropy: #store coefficients from box with maximum entropy
                        maxEntropy = entropy(abs(localBox.flatten()))
                        localboxCos = [i-int(boxHeight/2),i+int(boxHeight/2),j,j+boxWidth]
                        threses2[len(threses2)-1] = 4.5*np.std(localBox) #thres is 4.5 x standard deviation of 'noise' to cover 99.99% of noise

                horStrips[len(horStrips)-1] = localboxCos

            #horizontal threshold pass
            for i,coeff in enumerate(coeffs):
                for j,val in enumerate(coeff):

                    foundBox = False
                    for k in range(len(horStrips)):

                        if (i >= horStrips[k][0]) and (i <= horStrips[k][1]):
                            thres = threses2[k]
                            foundBox = True

                    if not foundBox: 
                        thres = 0
                        # print('oops')
                        print(i*(sampleRate/2)/len(coeffs))

                    if abs(val) < thres:
                        coeff[j] = coeff[j]*0.4 #soft thres-holding

            # Apply thresholding to all coeffs (vertical pass)
            for i,coeff in enumerate(coeffs):
                for j,val in enumerate(coeff):

                    foundBox = False
                    for k in range(len(vertStrips)):

                        if (j >= vertStrips[k][2]) and (j <= vertStrips[k][3]):
                            thres = threses[k]
                            foundBox = True

                    if not foundBox: 
                        thres = 0
                        # print('oops')
                        print(j*(len(signal)/sampleRate)/len(coeffs[0]))

                    if abs(val) < thres:
                        coeff[j] = coeff[j]*0.4 #soft thres-holding
            
            #thresholding for low frequencies (<800 Hz)
            for i,coeff in enumerate(coeffs):
                for j,val in enumerate(coeff):
                    if i*(sampleRate/2)/len(coeffs) < 900:
                        coeff[j] = coeff[j]*0.1

            specGraphBoxes = []
            for box in vertStrips:
                boxCos = [(sampleRate/2)*box[0]/len(coeffs),(sampleRate/2)*box[1]/len(coeffs),(len(signal)/sampleRate)*box[2]/len(coeffs[0]),(len(signal)/sampleRate)*box[3]/len(coeffs[0])]
                specGraphBoxes.append(boxCos)
            specGraphBoxes2 = []
            for box in horStrips:
                boxCos = [(sampleRate/2)*box[0]/len(coeffs),(sampleRate/2)*box[1]/len(coeffs),(len(signal)/sampleRate)*box[2]/len(coeffs[0]),(len(signal)/sampleRate)*box[3]/len(coeffs[0])]
                specGraphBoxes2.append(boxCos)
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

    return signal, [specGraphBoxes,specGraphBoxes2]


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


# Noisy possum Test
sampleRate, signal = wave.read('recordings/PossumNoisy.wav') # possum.wav works well Haar or dmey, 5, partial, thres=96*4.5
form = signal.dtype
wavelet = 'dmey'
print(sampleRate)

level = pywt.dwt_max_level(len(signal), wavelet) # Calculate the maximum level
# level = 9
# Normalisation
# signal = (signal - np.mean(signal)) / np.std(signal) # Normalisation
# signal = np.divide(signal, sum(signal)) # Generate random variable that add to 1.0

# print(pywt.wavelist(kind='discrete'))
print(f"Max level: {level}")

thres = findThres(signal, level)
# thres = universalThresholding(signal)
print(thres)
# decomposition(signal, level)


denoised, specGraphBoxes = thresholdFull(signal, thres, wavelet=wavelet, levels=level)
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

#display boxes of noise selection
for box in specGraphBoxes[0]: #vertical pass
    box = np.array([[box[2],box[0]],[box[3],box[0]],[box[3],box[1]],[box[2],box[1]]])
    p = Polygon(box, fill=False, color='black')
    ax1.add_patch(p)

for box in specGraphBoxes[1]:#horizontal pass
    box = np.array([[box[2],box[0]],[box[3],box[0]],[box[3],box[1]],[box[2],box[1]]])
    p = Polygon(box, fill=False, color='red')
    ax1.add_patch(p)

plt.show()

# Save denoised signal
wave.write('denoised/denoised.wav', sampleRate, denoised)
