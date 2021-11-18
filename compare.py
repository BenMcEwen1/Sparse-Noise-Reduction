import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import os


filename = 'denoised/possum44k'

sampleRate, s = wave.read(f'recordings/{filename}.wav')

directory = 'reference/original'


def extract(directory):
    # Extract recordings, repalce with SD card directory
    masks = []
    calls = []

    for dirpaths, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            data = np.load(f"{directory}/{filename}")
            calls.append(filename)
            masks.append(data)

    return masks, calls


def spectrograms(recording, sampleRate, plot=False):
    # Plot spectrograms of recording and ref
    fp, tp, Sp = signal.spectrogram(recording, fs=sampleRate)

    e = 1e-11
    Sp = np.log(Sp + e)

    if plot:
        cmap = plt.get_cmap('magma')
        plt.pcolormesh(tp, fp, Sp, cmap=cmap, shading='auto')
        plt.show()

    return Sp


def correlation(recording, masks, sampleRate):
    # Convolve spectrogram with ref to generate correlation
    Sp = spectrograms(recording, sampleRate)

    cor = []
    scaled = []

    lower = 0
    upper = 0

    for mask in masks:
        c = signal.convolve2d(Sp, mask, mode="valid", boundary="wrap")
        c = abs(np.subtract(c[0],max(c[0])))  

        if c.min() < lower:
            lower = c.min()
        if c.max() > upper:
            upper = c.max()
        
        cor.append(c)

    # Scale correlation relative to upper and lower values
    for c in cor:
        c = np.interp(c, (lower,upper), (0,1)) 
        scaled.append(c)

    return scaled


def correlation2(recording, ref, sampleRate):
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


def findRegions(correlation, threshold=0.5):
    # Find the regions of interest
    regions = []
    for cor in correlation:
        recommend = []
        for c in cor:
            if c >= threshold:
                recommend.append(1) # append highest correlation for that region
            else:
                recommend.append(0)

        regions.append(dilation(recommend))

    return regions 


def segment(cor, mask):
    # Segment regions of interest
    seg = []
    for i,m in enumerate(mask):
        seg.append(np.multiply(cor[i], m))
    return seg


def extractTimeStamp(masks, sampleRate):
    # Return time stamp of regions of interest
    
    stamp = []
    for mask in masks:
        state = mask[0]
        times = []
        for i,m in enumerate(mask):
            if m != state:
                times.append(i)
                state = m
        stamp.append(times)

    return stamp


def save(signal, sampleRate, stamp, filename):
    # Save segmented signal
    stamp = np.multiply(stamp,238.5)

    for i in range(0, len(stamp), 2):
        seg = signal[int(stamp[i]):int(stamp[i+1])]
        wave.write(f'segmented/{filename}_{i}.wav', sampleRate, seg)


def rank(correlation, stamps):
    # Rank in order of highest correlation
    rs = []
    for i,stamp in enumerate(stamps):

        cor = correlation[i]
        r = []
        for j in range(0,len(stamp),2):
            maxCorrelation = max(cor[int(stamp[j]):int(stamp[j+1])])
            r.append(((stamp[j],stamp[j+1]), maxCorrelation))
        
        rs.append(r)

    return rs

# Extract masks
masks, calls = extract(directory)

norm = []
for mask in masks:
    # mask = np.subtract((mask / mask.max()), mask.max()/2)
    mask = np.subtract(mask, mask.max()/2)
    norm.append(mask)

masks = norm

# For a given field recording and array of masks generate array of correlations
cor = correlation(s, masks, sampleRate)

for i,c in enumerate(cor):
    print(calls[i])
    plt.plot(c)
    plt.ylim(0,1.2)
    plt.show()

# Extract regions of interest
regions = findRegions(cor)

# Segment regions of interest given regions
seg = segment(cor, regions)

# Extract time stamps
stamp = extractTimeStamp(regions, sampleRate)

# Display correlation/rank with relevant time stamp
r = rank(cor, stamp)

# Combine
def combine(calls, r):
    com = []
    for i in range(len(calls)):
        com.append((calls[i], r[i]))
    return com

print(combine(calls,r))