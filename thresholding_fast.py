import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
from scipy.stats import entropy, kstest, uniform
import matplotlib.pyplot as plt
import time
from math import ceil, floor
from matplotlib.patches import Polygon

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


def getBoxes(coeffs, decompositionState='full', numboxes=40): #numboxes is number of boxes horizontally, if full decomp. this is repeated for vertical boxes
    #returns box coordinates in coefficient-leaf form, st. deviation information of boxes

    print(coeffs.shape)

    if decompositionState == 'full':
        numboxesV = numboxes
        numboxesH = numboxes

        edge = 0

        boxWidth = int((len(coeffs[0])-edge*2)/numboxesH)
        boxHeight = int(len(coeffs)/numboxesV)
        numboxesV = floor(len(coeffs)/boxHeight)
        numboxesH = floor((len(coeffs[0])-edge*2)/boxWidth)
        
        print(f'{numboxesH} boxes {boxWidth} wide, {numboxesV} boxes {boxHeight} tall')

        boxVals = np.empty((numboxesV, numboxesH, boxHeight, boxWidth))
        boxCos = np.empty((numboxesV, numboxesH, 4))

        for i in range(numboxesV):
            for j in range(numboxesH):
                boxVals[i,j] = np.array(coeffs[boxHeight*i:boxHeight*(i+1),boxWidth*j:boxWidth*(j+1)])
                boxCos[i,j] = np.array([boxHeight*i,boxHeight*(i+1),boxWidth*j,boxWidth*(j+1)])

        maxRowSDs = []
        minRowSDs = []
        boxSDs = np.empty((numboxesV, numboxesH))

        for i in range(len(boxVals)): #iterating one row at a time
            for j in range(len(boxVals[0])): #iterating through each value in row
                
                boxSDs[i,j] = np.std(boxVals[i,j].flatten()) #standard deviation of coefficients within box i,j

            # print(f'row {i} has SDs of {boxSDs[i]}')
            maxRowSDs.append(max(boxSDs[i])) 
            minRowSDs.append(min(boxSDs[i])) 
            # print(f'max SD in row {i} is {maxRowSDs[i]}')

        print(f'boxCos: {boxCos.shape}')
        
        return boxCos, boxSDs, minRowSDs, maxRowSDs
        

def thresholdFull(signal, wavelet='dmey', levels=5):

    coeffs = decomposeFull(signal, wavelet=wavelet, levels=levels, plot=False)

    print(f"Number of leaves: {len(coeffs)}, Coeffs per leaf: {len(coeffs[0])}")
    
    coeffs = np.array(coeffs)
    boxWidth = int(sampleRate*len(coeffs[0])/len(signal))

    # thresholding for low frequencies (<600 Hz)
    for i,coeff in enumerate(coeffs):
        for j,val in enumerate(coeff):
            if (i*(sampleRate/2)/len(coeffs) < 800) and (len(coeffs)>30): # have to check there is sufficient frequency resolution to discard low frequencies
                coeff[j] = coeff[j]*((i*(sampleRate/2)/len(coeffs))/800)**3
    
    print("Get boxes")
    boxCos, boxSDs, minRowSDs, maxRowSDs = getBoxes(coeffs, decompositionState='full', numboxes=40)

    fig, ax = plt.subplots(1)
    ax.plot(boxSDs)

    fig, ax = plt.subplots(1)
    ax.plot(np.array(maxRowSDs)-np.array(minRowSDs),range(len(boxCos)))

    rowVariation = np.array(maxRowSDs)-np.array(minRowSDs)

    SDthreshold = 1/2 #boxes with SD greater than 1/8 of the maximum SD are assumed to be signal and not thresholded
    SDvariationThres = 1/2 #rows with a SD variation greater than 1/7 of the max with not be thresholded

    for ii, row in enumerate(boxCos):
        
        thres = minRowSDs[ii]*8.5 #threshold is 4.5 * SD of the lowest SD box in the row
        softness = 0.1
        if rowVariation[ii] < max(rowVariation)*SDvariationThres:
            thres = thres*0.5

        for jj, box in enumerate(boxCos[ii]):

            if boxSDs[ii,jj] < np.max(maxRowSDs)*SDthreshold:
                for i in range(int(boxCos[ii,jj,0]),int(boxCos[ii,jj,1])):
                    for j in range(int(boxCos[ii,jj,2]),int(boxCos[ii,jj,3])):
                        if coeffs[i,j] < thres:
                            coeffs[i,j] *= softness

    # Reconstruct each level
    signal = reconstructFull(coeffs, wavelet=wavelet, plot=False) # Full tree reconstruction

    return signal

def thresholdPartial(signal, wavelet='dmey', levels=5):

    coeffs = pywt.wavedec(signal, wavelet='dmey',mode='symmetric', level=levels)

    specGraphBoxes = [[],[]]
    boxes = []
    boxEs = []
    boxStds = []
    rowVariance = []
    rowminStd = []

    for i,coeff in enumerate(coeffs):
        print(f'layer {i}, time coeffs: {len(coeff)}')
        specGraphBoxes.append([])

        boxwidth = int(len(coeff)/80)
        numboxes = floor(len(coeff)/boxwidth)

        if ((sampleRate/2) * 1/(2**(levels-i+1))) < 600: #threshold low freqencies
            coeff *= 0.2

        rowBegin = len(boxEs)

        for j in range(0,len(coeff)-boxwidth,boxwidth):
            boxvals = coeff[j:j+boxwidth]
            boxes.append([i,j,j+boxwidth])
            boxEs.append(entropy(abs(boxvals)))
            boxStds.append(np.std(boxvals))

        rowVariance.append(max(boxStds[rowBegin:])-min(boxStds[rowBegin:]))
        rowminStd.append(min(boxStds[rowBegin:]))

    maxE = max(np.array(boxEs))
    minE = min(np.array(boxEs))
    maxStd = max(np.array(boxStds))
    minStd = min(np.array(boxStds))

    count = 0
    for k,box in enumerate(boxes):
        thres = rowminStd[box[0]]*4.5
        softness = (rowVariance[box[0]]-min(rowVariance))/(max(rowVariance)-min(rowVariance))

        for j in range(box[1],box[2]):
            if coeffs[box[0]][j] < thres:
                coeffs[box[0]][j] *= softness


    #frequency bands
    freqs = np.zeros(levels+2)
    for i in range(levels+1):
        freqs[i] = (sampleRate/2) * 1/(2**(levels-i+1))
    freqs[0] = 0
    freqs[levels+1] = sampleRate/2

    #make coordinates for boxes on spectogram
    for i,box in enumerate(boxes):
        relativeStd = (boxStds[i]-minStd)/(maxStd-minStd)
        relativeE = (boxEs[i]-minE)/(maxE-minE)

        cos = np.zeros(4)
        cos[0] = freqs[box[0]]
        cos[1] = freqs[box[0]+1]

        cos[2] = ((box[1])/(len(coeffs[box[0]])))*(len(signal)/sampleRate)
        cos[3] = ((box[2])/(len(coeffs[box[0]])))*(len(signal)/sampleRate)

        rect = [[cos[2],cos[0]],[cos[3],cos[0]],[cos[3],cos[1]],[cos[2],cos[1]]]
        rect0 = Polygon(rect, fill=True, color=[relativeE,relativeE,relativeE])
        specGraphBoxes[0].append(rect0)
        rect1 = Polygon(rect, fill=True, color=[relativeStd,relativeStd,relativeStd])
        specGraphBoxes[1].append(rect1)

    signal = pywt.waverec(coeffs, wavelet='dmey')

    return signal, specGraphBoxes

sampleRate, signal = wave.read('recordings/original/PossumNoisy.wav') # possum.wav works well Haar or dmey, 5, partial, thres=96*4.5
form = signal.dtype
wavelet = 'dmey'
print(sampleRate)

level = pywt.dwt_max_level(len(signal), wavelet)

level = 8

print(f"Max level: {level}")

start = time.time()
denoised = thresholdFull(signal, wavelet=wavelet, levels=level)
# denoised, specGraphBoxes = thresholdPartial(signal, wavelet=wavelet, levels=level)


print(f'Time taken: {time.time()-start} seconds')

plt.figure()
plt.title('Original/Denoised signal')
plt.plot(signal,color='black')
plt.plot(denoised)

fig, ([ax1, ax2]) = plt.subplots(2)
fig.suptitle('Original/Denoised Spectrogram')
ax1.specgram(signal, Fs=sampleRate)
denoised = np.asarray(denoised, dtype=form) # Downsample
ax2.specgram(denoised, Fs=sampleRate)
# ax3.specgram(signal, Fs=sampleRate)
# ax4.specgram(signal, Fs=sampleRate)
# ax4.set_title('standard deviations')
# ax3.set_title('entropy')
ax2.set_title('denoised')

# for box in specGraphBoxes[0]:#horizontal boxes
#     ax3.add_patch(box)
# for box in specGraphBoxes[1]:#horizontal boxes
#     ax4.add_patch(box)

plt.show()

# Save denoised signal
wave.write('denoised/denoised.wav', sampleRate*int(len(denoised)/len(signal)), denoised)
