import librosa
import math
import numpy as np


def SNR(original, denoised):
    S = sum(denoised**2)/len(denoised)
    N = sum(original**2)/len(original)
    return 10*math.log10(S/N)

def SnNR(signal, noise):
    S = sum(signal**2)/len(signal)
    N = sum(noise**2)/len(noise)
    return 10*math.log((S + N)/N)

def success_ratio(original, denoised):
    return math.log10(np.var(original)/np.var(denoised))

def PSNR(original, denoised):
    MAX = np.max(original)
    MSE = np.mean((original - denoised)**2)
    return 20*math.log10(MAX/math.sqrt(MSE))

filename = "feature.wav"
orign = "Mix-4.wav"
noise, sr = librosa.load('./feature_spacing/noise.wav')
clean, sr = librosa.load(f'./feature_spacing/{filename}')
original, sr = librosa.load(f'./feature_spacing/original/{orign}')

SnNR_score = SnNR(clean, noise)
SR_score = success_ratio(original[0:(5*sr)], clean[0:(5*sr)])
PSNR_score = PSNR(original[0:(5*sr)], clean[0:(5*sr)])

print(filename)
print(SnNR_score)
print(SR_score)
print(PSNR_score)

