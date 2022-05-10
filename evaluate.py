import librosa
import math
import numpy as np
import matplotlib.pyplot as plt
import json


def SNR(original, denoised):
    S = sum(denoised**2)/len(denoised)
    N = sum(original**2)/len(original)
    return 10*math.log10(S/N)

def SnNR(original, denoised, sigRange, noiseRange):
    denoised = denoised[sigRange[0]:sigRange[1]]
    original = original[noiseRange[0]:noiseRange[1]]
    S = sum(denoised**2)/len(denoised)
    N = sum(original**2)/len(original)
    return 10*math.log((S + N)/N)

def success_ratio(original, denoised, noiseRange): # WORKING
    denoised = denoised[noiseRange[0]:noiseRange[1]]
    original = original[noiseRange[0]:noiseRange[1]]
    return math.log10(np.var(original)/np.var(denoised))

def PSNR(original, denoised): # WORKING
    MAX = np.max(original)
    MSE = np.mean((original - denoised)**2)
    return 20*math.log10(MAX/math.sqrt(MSE))


def dataLoader(plot=False):
    with open("dataset.json", "r") as dataset:
        data = json.load(dataset)

    for filename in data.keys():
        if filename:
            results = {}
            original, sampleRate = librosa.load(f'./audio/{filename}.wav', sr=None)
            denoised, sampleRate = librosa.load(f'./denoised_SNG/{filename}_denoised.wav', sr=None)
            original = original[:len(denoised)] # Ensure signals are the same length

            signalStart = data[filename]['signalStart'] * sampleRate
            signalEnd = data[filename]['signalEnd'] * sampleRate
            noiseStart = data[filename]['noiseStart'] * sampleRate
            noiseEnd = data[filename]['noiseEnd'] * sampleRate

            SNR_db = SNR(original, denoised)
            SnNR_db = SnNR(original, denoised, [signalStart,signalEnd], [noiseStart,noiseEnd])
            sr = success_ratio(original, denoised, [noiseStart,noiseEnd]) # WORKING
            psnr = PSNR(original, denoised) # WORKING

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

            print(f'Results: {results}')

    return results

results = dataLoader(plot=False)