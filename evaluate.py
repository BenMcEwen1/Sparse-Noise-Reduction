import librosa
import math
import numpy as np
import matplotlib.pyplot as plt
import os


def SNR(signal, denoised):
    Psig = sum(signal**2)/len(signal)
    Pnoise = sum(denoised**2)/len(denoised)
    print(f'Noise Power: {10*math.log10(Pnoise)}')
    return 10 * math.log10(Psig/Pnoise)

def SnNR(signal, noise, sampleRate):
    signal = signal[0*sampleRate:7*sampleRate]
    noise = noise[8*sampleRate:9*sampleRate]
    Psig = sum(signal**2)/len(signal)
    Pnoise = sum(noise**2)/len(noise)
    return 10 * math.log10((Psig + Pnoise) / Pnoise)

def success_ratio(denoised, original, sampleRate):
    denoised = denoised[0*sampleRate:7*sampleRate]
    original = original[8*sampleRate:9*sampleRate]
    return math.log10(np.var(denoised)/np.var(original))

def PSNR(denoised, original):
    MSE = np.mean((denoised - original)**2)
    MAX = np.max(original)
    return 20 * math.log10(MAX/math.sqrt(MSE))


def dataLoader(filename, plot=False):
    results = {}
    original, sampleRate = librosa.load(f'./audio/{filename}.wav', sr=None)
    denoised, sampleRate = librosa.load(f'./denoised/{filename}_denoised.wav', sr=None)
    original = original[:len(denoised)]

    SNR_db = SNR(original, denoised)
    SnNR_db = SnNR(original, denoised, sampleRate)
    sr = success_ratio(original, denoised, sampleRate)
    psnr = PSNR(original, denoised)

    results[filename] = {
        'SNR_db': SNR_db,
        'SnNR_db': SnNR_db,
        'sr': sr,
        'psnr': psnr
    }

    if plot:
        plt.figure(1)
        plt.title('Original')
        plt.specgram(original, Fs=sampleRate)

        plt.figure(2)
        plt.title('denoised')
        plt.specgram(denoised, Fs=sampleRate)
        plt.show()

    return results

filename = 'Alarm1'
results = dataLoader(filename, plot=False)
print(f'Results: {results}')




