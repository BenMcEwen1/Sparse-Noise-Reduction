import librosa
import math
import numpy as np
import matplotlib.pyplot as plt
import json


def SNR(original, denoised):
    S = sum(denoised**2)/len(denoised)
    N = sum(original**2)/len(original)
    return 10*math.log10(S/N)

def SnNR(original, sigRange, noiseRange):
    signal = original[sigRange[0]:sigRange[1]]
    noise = original[noiseRange[0]:noiseRange[1]]
    S = sum(signal**2)/len(signal)
    N = sum(noise**2)/len(noise)
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

    snnr_array = []
    sr_array = []
    psnr_array = []

    with open("Predator_dataset.json", "r") as dataset:
        data = json.load(dataset)

    for filename in data.keys():
        if filename:
            results = {}
            try:
                original, sampleRate = librosa.load(f'./rms/rms5/original/{filename}.wav', sr=None)
                denoised, sampleRate = librosa.load(f'./rms/rms5/denoised/{filename}.wav', sr=None)
                original = original[:len(denoised)] # Ensure signals are the same length

                # print(filename)

                signalStart = int(data[filename]['signalStart'] * sampleRate)
                signalEnd = int(data[filename]['signalEnd'] * sampleRate)
                noiseStart = int(data[filename]['noiseStart'] * sampleRate)
                noiseEnd = int(data[filename]['noiseEnd'] * sampleRate)

                SNR_db = SNR(original, denoised)

                # original
                SnNR_original = SnNR(original, [signalStart,signalEnd], [noiseStart,noiseEnd])
                SnNR_denoised = SnNR(denoised, [signalStart,signalEnd], [noiseStart,noiseEnd])

                sr = success_ratio(original, denoised, [noiseStart,noiseEnd]) # WORKING
                psnr = PSNR(original, denoised) # WORKING

                snnr_array.append(SnNR_denoised)
                sr_array.append(sr)
                psnr_array.append(psnr)

                results[filename] = {
                    # 'SNR_db': SNR_db,
                    'SnNR_original': SnNR_original,
                    'SnNR_denoised': SnNR_denoised,
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

                # print(f'Results: {results}')
                for call in results:
                #     # print(f"{results[call]['SnNR_original']}")
                    print(f"{results[call]['SnNR_denoised']},{results[call]['sr']},{results[call]['psnr']}")

            except:
                print(f'Could not find {filename}.wav')

    print(np.mean(snnr_array))
    print(np.mean(sr_array))
    print(np.mean(psnr_array))

    return results

results = dataLoader(plot=False)