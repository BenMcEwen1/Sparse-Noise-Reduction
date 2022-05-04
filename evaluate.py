import librosa
import math
import numpy as np
import matplotlib.pyplot as plt

filename = 'AustralasianBittern_5min'
recording, sample_rate = librosa.load(f'./Sound Files/{filename}.wav', sr=None)

filename = 'AustralasianBittern_5min_denoised'
denoised, sample_rate = librosa.load(f'./processed/{filename}.wav', sr=None)

recording = recording[0:2300000]
denoised = denoised[0:2300000]


def SNR(signal, denoised):
    Psig = sum(signal**2)/len(signal)
    Pnoise = sum(denoised**2)/len(denoised)

    return 10 * math.log10(Psig/Pnoise)

def SnNR(signal, noise):
    signal = signal[60*sample_rate:2*60*sample_rate]
    noise = noise[0:30*sample_rate]

    Psig = sum(signal**2)/len(signal)
    Pnoise = sum(noise**2)/len(noise)

    return 10 * math.log10((Psig + Pnoise) / Pnoise)

def success_ratio(denoised, original):
    denoised = denoised[0:30*sample_rate]
    original = original[0:30*sample_rate]

    return math.log10(np.var(denoised)/np.var(original))

def PSNR(denoised, original):
    MSE = np.mean((denoised - original)**2)
    MAX = np.max(original)

    return 20 * math.log10(MAX/math.sqrt(MSE))


SNR_db = SNR(recording, denoised)
SnNR_db = SnNR(recording, denoised)
sr = success_ratio(recording, denoised)
psnr = PSNR(recording, denoised)

print('-------')
print(f'Signal to Noise Ratio (SNR): {SNR_db}')
print(f'SnNR: {SnNR_db}')
print(f'Success Ratio: {sr}')
print(f'Peak Signal Noise Ratio (PSNR): {psnr}')
print('-------')

plt.figure(1)
plt.title('Original')
plt.specgram(recording, Fs=sample_rate)

plt.figure(2)
plt.title('denoised')
plt.specgram(denoised, Fs=sample_rate)
plt.show()