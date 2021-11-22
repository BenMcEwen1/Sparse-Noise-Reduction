import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
from scipy.stats import entropy, kstest, uniform
import matplotlib.pyplot as plt
import time
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


def thresholdFull(signal, wavelet='dmey', levels=5):
    e = entropy(abs(signal))
    stop = False

    for l in range(levels,levels+1): # For final leaf nodes set the start range = levels
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
        
        #box search method
        if (l == level) or (stop==True): #only thresholding at the lowest level
            print(f"Level {l}")
            print(f"Number of leaves: {len(coeffs)}, Coeffs per leaf: {len(coeffs[0])}")
            
            # search through coefficient array with boxes to find highest entropy region, treat this as noise
            threses = []
            horStrips = []
            boxEntropies = []
            boxStds = []

            #boxwidth is 1 second
            boxWidth = int(sampleRate*len(coeff)/len(signal))
            boxHeight = int(len(coeffs)/40)*2

            #find boxes
            for i in range(int(boxHeight/2), len(coeffs), int(boxHeight/2)):
                horStrips.append([])
                threses.append([])
                boxEntropies.append([])
                boxStds.append([])

                maxEntropy = 0

                for j in range(0, len(coeffs[0])-int(boxWidth/2), int(boxWidth/2)): #iterate over array with stepsize of 1/2 the box width/height

                    localBox = np.array(coeffs)[i-int(boxHeight/2):i+int(boxHeight/2), j:j+boxWidth]

                    if entropy(abs(localBox.flatten())) > maxEntropy: #store coefficients from box with maximum entropy

                        maxEntropy = entropy(abs(localBox.flatten()))
                        localboxCos = [i-int(boxHeight/2),i+int(boxHeight/2),j,j+boxWidth]
                        threses[len(threses)-1] = 4.5*np.std(localBox) #thres is 4.5 x standard deviation of 'noise' to cover 99.99% of noise
                        boxEntropies[len(boxEntropies)-1] = maxEntropy
                
                horStrips[len(horStrips)-1] = localboxCos
            

            #horizontal strip threshold pass
            for i,coeff in enumerate(coeffs):
                if entropy(abs(coeff)) > (0.5*(max(currentE)-min(currentE)) + min(currentE)):
                    for j,val in enumerate(coeff):

                        foundBox = False
                        for k in range(len(horStrips)):

                            if (i >= horStrips[k][0]+boxHeight/4) and (i <= horStrips[k][1]-boxHeight/4):
                                thres = threses[k]
                                foundBox = True

                        if not foundBox:
                            for k in range(len(horStrips)):
                                if (i >= horStrips[k][0]) and (i <= horStrips[k][1]):
                                    thres = threses[k]
                                    foundBox = True
                            
                            if not foundBox: #(still)
                                thres = 0
                                print(i*(sampleRate/2)/len(coeffs))#print frequency of coefficient wihtout box
                                softness = 1

                        if abs(val) < thres:
                            coeff[j] = coeff[j]*0.1 #soft thres-holding
            
            #thresholding for low frequencies (<600 Hz)
            for i,coeff in enumerate(coeffs):
                for j,val in enumerate(coeff):
                    if (i*(sampleRate/2)/len(coeffs) < 300) and (len(coeffs)>30):#have to check there is sufficient frequency resolution to discard low frequencies
                        coeff[j] = coeff[j]*((i*(sampleRate/2)/len(coeffs))/300)**2
            
            #create box coordinates for spectrogram display
            specGraphBoxes = []
            for i,box in enumerate(horStrips):
                box[0] = len(coeffs)-box[0]
                box[1] = len(coeffs)-box[1]
                rect = [(sampleRate/2)*box[0]/len(coeffs),(sampleRate/2)*box[1]/len(coeffs),(len(signal)/sampleRate)*box[2]/len(coeffs[0]),(len(signal)/sampleRate)*box[3]/len(coeffs[0])]
                rect = np.array([[rect[2],rect[0]],[rect[3],rect[0]],[rect[3],rect[1]],[rect[2],rect[1]]])
                rect = Polygon(rect, fill=False, color=[0,0,0])
                specGraphBoxes.append(rect)

        # Reconstruct each level
        signal = reconstructFull(coeffs, wavelet=wavelet, plot=False) # Full tree reconstruction

        if stop:
            print('stopping due to entropy condition')
            break

    return signal, specGraphBoxes


def thresholdPartial(signal, thres, wavelet='dmey', level=5):
    coeffs = partialTree(signal, levels=level, plot=False)

    # for i,coeff in enumerate(coeffs):
    #     if i != 0: # First index is the Approximate node, careful!
    #         thres = thres * 0.6 # AviaNZ suggests the std of the lowest packet x4.5
    #         # thres = universalThresholding(coeff)
    #         coeffs[i] = pywt.threshold(coeff, value=thres, mode='soft') # 0.2 works well for chirp

    specGraphBoxes = []

    for i,coeff in enumerate(coeffs):
        print(i)
        print(len(coeff))
        specGraphBoxes.append([])
        boxwidth = int(sampleRate*len(coeff)/len(signal))
        #search for highest entropy section
        highestEntropy = 0

        for j in range(0,len(coeff)-boxwidth,int(boxwidth/4)):

            box = coeff[j:j+boxwidth]

            if entropy(abs(box))>highestEntropy:
                highestEntropy = entropy(abs(box))
                thres = np.std(box)
                specGraphBoxes[len(specGraphBoxes)-1]=[len(signal)*j/(len(coeff)*sampleRate),len(signal)*(j+boxwidth)/(len(coeff)*sampleRate)]
        for j,val in enumerate(coeff):

            if val>thres:
                coeff[j] = coeff[j]*0.2#soft

        # if i<=5:#len(coeffs)/2:
        #     for j,val in enumerate(coeff):
        #         if j>len(coeff)/2:
        #             coeff[j] = coeff[j]*0


    signal = pywt.waverec(coeffs, wavelet='dmey')

    return signal, specGraphBoxes


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


def evaluateMethod(signal,denoised,noiseTime): #aviaNZ's evaluation metrics
    #signal to noise ratio SNR
    signalPower = sum(signal**2)/len(signal)
    noise = signal[int(noiseTime[0]*sampleRate):int(noiseTime[1]*sampleRate)]
    noisePower = sum(noise**2)/len(noise)

    denoisedSignalPower = sum(denoised**2)/len(denoised)
    denoisedNoise = denoised[int(noiseTime[0]*sampleRate):int(noiseTime[1]*sampleRate)]
    denoidedNoisePower = sum(denoisedNoise**2)/len(denoisedNoise)

    SnNR = 10*np.log10(signalPower/noisePower)
    SnNR2 = 10*np.log10(denoisedSignalPower/denoidedNoisePower)
    print(f'\nOriginal signal to noise ratio (dB): {SnNR}')
    print(f'Denoised signal to noise ratio (dB): {SnNR2}')
    print(f'\nImprovement (dB): {SnNR2-SnNR}')

    #Success ratio
    print(f'Success Ratio: {np.log10(np.var(noise)/np.var(denoisedNoise))}')

    #Peak signal to noise ratio PSNR
    meanSqaureError = np.mean((signal-denoised[0:len(signal)])**2)
    PSNR = 10*np.log10((max(signal)**2)/meanSqaureError)
    print(f'Peak signal to noise ratio: {PSNR}')

# Noisy possum Test
sampleRate, signal = wave.read('recordings/PossumNoisy.wav') # possum.wav works well Haar or dmey, 5, partial, thres=96*4.5
form = signal.dtype
wavelet = 'dmey'
print(sampleRate)

level = pywt.dwt_max_level(len(signal), wavelet) # Calculate the maximum level

# print(pywt.wavelist(kind='discrete'))
print(f"Max level: {level}")

start = time.time() #check time taken
denoised, specGraphBoxes = thresholdFull(signal, wavelet=wavelet, levels=level)
# denoised, specGraphBoxes = thresholdPartial(signal, thres, wavelet=wavelet, level=level)

print(f'Time taken: {time.time()-start} seconds')

evaluateMethod(signal,denoised,[0,0.2])

plt.figure()
plt.title('Original/Denoised signal')
plt.plot(signal,color='black')
plt.plot(denoised)
plt.show()

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Original/Denoised Spectrogram')
ax1.specgram(signal, Fs=sampleRate)
denoised = np.asarray(denoised, dtype=form) # Downsample
ax2.specgram(denoised, Fs=sampleRate)

for box in specGraphBoxes:#horizontal boxes
    ax1.add_patch(box)

plt.show()

# Save denoised signal
wave.write('denoised/denoised.wav', sampleRate*int(len(denoised)/len(signal)), denoised)
