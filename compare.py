import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt


filename = 'possum_clean'

sampleRate, s = wave.read(f'recordings/{filename}.wav')
sampleRate, ref = wave.read('recordings/possum_snip.wav')


def spectrograms(recording, ref, sampleRate, plot=False):
    # Plot spectrograms of recording and ref
    fp, tp, Sp = signal.spectrogram(recording, fs=sampleRate)
    fr, tr, Sr = signal.spectrogram(ref, fs=sampleRate)

    e = 1e-11
    Sp = np.log(Sp + e)
    Sr = np.log(Sr + e)

    if plot:
        cmap = plt.get_cmap('magma')
        plt.subplot(1,2,1)
        plt.pcolormesh(tp, fp, Sp, cmap=cmap)
        plt.subplot(1,2,2)
        plt.pcolormesh(tr, fr, Sr, cmap=cmap)
        plt.show()

    return Sp, Sr


def correlation(recording, ref, sampleRate):
    # Convolve spectrogram with ref to generate correlation
    Sp, Sr = spectrograms(recording, ref, sampleRate, plot=True)

    cor = signal.convolve2d(Sp, Sr, mode="valid", boundary="wrap")

    cor = abs(np.subtract(cor[0],max(cor[0])))   
    cor = np.interp(cor, (cor.min(),cor.max()), (0,1)) 

    return cor


def dilation(recommend, k=50):
    # Expand binary mask to include surrounding areas
    d = []
    for i in range(len(recommend)):
        if any(recommend[i-k:i+k]) == 1:
            d.append(1)
        else:
            d.append(0)
    return d


def findRegions(correlation, threshold=0.4):
    # Find the regions of interest
    recommend = []
    for c in correlation:
        if c >= threshold:
            recommend.append(1)
        else:
            recommend.append(0)

    return dilation(recommend)


def segment(signal, mask):
    # Segment regions of interest
    return np.multiply(signal, mask)


def extractTimeStamp(mask, sampleRate):
    # Return time stamp of regions of interest
    state = mask[0]
    stamp = []
    for i,m in enumerate(mask):
        if m != state:
            stamp.append(i)
            state = m

    # convert to time domain
    stamp = np.multiply(stamp,238.5)
    return stamp


def save(signal, sampleRate, stamp, filename):
    # Save segmented signal
    for i in range(0, len(stamp), 2):
        seg = signal[int(stamp[i]):int(stamp[i+1])]
        wave.write(f'segmented/{filename}_{i}.wav', sampleRate, seg)



cor = correlation(s, ref, sampleRate)
mask = findRegions(cor)
seg = segment(cor, mask)

stamp = extractTimeStamp(mask, sampleRate)

save(s, sampleRate, stamp, filename)