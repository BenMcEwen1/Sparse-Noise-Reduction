import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import os


filename = 'downsampled/stoat16k' # Careful! signal and reference must have the same sample rates

sampleRate, s = wave.read(f'recordings/{filename}.wav')

directory = 'reference/original'


def extract(directory):
    # Extract recordings, repalce with SD card directory
    masks = []
    calls = []

    for dirpaths, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            data = np.load(f"{directory}/{filename}")
            calls.append(filename[0:-4])
            masks.append(data)

    return masks, calls


def spectrograms(recording, sampleRate, plot=False):
    # Plot spectrograms of recording and ref
    fp, tp, Sp = signal.spectrogram(recording, fs=sampleRate)

    if plot:
        cmap = plt.get_cmap('magma')
        plt.pcolormesh(tp, fp, Sp, cmap=cmap, shading='auto')
        plt.show()

    return Sp


def normalise(mask):
    # Normalise to prevent higher energy masks becoming biased
    norm = np.linalg.norm(mask)
    mask = np.divide(mask, norm)
    mask = mask / mask.sum()
    return mask


def correlation(recording, masks, sampleRate):
    # Convolve spectrogram with ref to generate correlation
    Sp = spectrograms(recording, sampleRate)

    # Normalisation
    Sp = normalise(Sp)

    kernel = np.ones((2,2)) * 0.5

    cor = []
    scaled = []

    lower = 0
    upper = 0

    for mask in masks:
        # Normalise Mask
        mask = normalise(mask)

        # Smoothing (Optional)
        mask = signal.convolve2d(mask, kernel, mode='same', boundary='wrap', fillvalue=0)

        c = signal.correlate(Sp, mask, mode="valid")

        if c.min() < lower:
            lower = c.min()
        if c.max() > upper:
            upper = c.max()
        
        cor.append(c[0])

    # Scale correlation relative to upper and lower values
    for c in cor:
        c = np.interp(c, (lower,upper), (0,1)) 
        scaled.append(c)

    return scaled


def dilation(recommend, k=20):
    # Expand binary mask to include surrounding areas
    d = []
    for i in range(len(recommend)):
        if any(recommend[i-k:i+k]) == 1:
            d.append(1)
        else:
            d.append(0)
    return d


def findRegions(correlation, threshold=0.2):
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


def rank(correlation, stamps, calls):
    # Rank in order of highest correlation
    rs = []
    for i,stamp in enumerate(stamps):

        # If length of list is odd add timestamp to end
        if (len(stamp) % 2):
            stamp.append(len(correlation[i]))

        cor = correlation[i]
        r = []
        for j in range(0,len(stamp),2):
            
            maxCorrelation = max(cor[int(stamp[j]):int(stamp[j+1])])
            r.append((calls[i], (stamp[j],stamp[j+1]), maxCorrelation))
            # print(r)
        # rs.append(r)
        rs.extend(r)

    # Order in terms of correlation
    rs = sorted(rs, key=lambda x: x[2], reverse=True)

    return rs


# Extract masks
masks, calls = extract(directory)

# For a given field recording and array of masks generate array of correlations
cor = correlation(s, masks, sampleRate)

# for i,c in enumerate(cor):
#     print(calls[i])
#     plt.plot(c)
#     plt.ylim([0,1.1])
#     plt.show()

# Extract regions of interest
regions = findRegions(cor)

seg = segment(cor, regions)
# for i,s in enumerate(seg):
#     print(calls[i])
#     plt.plot(s)
#     plt.show()

# Extract time stamps
stamp = extractTimeStamp(regions, sampleRate)

# Display correlation/rank with relevant label and time stamp
r = rank(cor, stamp, calls)

print(r)