from scipy.io import wavfile
import noisereduce as nr
import os, random

NOISY_PATH = './rms/rms5/original/'
CLEAN_PATH = './rms/rms5/denoised/'

def denoise():
    for _,_,filenames in os.walk(NOISY_PATH):
        for filename in sorted(filenames):
            print(filename)
            rate, data = wavfile.read(f"{NOISY_PATH}{filename}")
            reduced_noise = nr.reduce_noise(y=data, sr=rate)
            wavfile.write(f"{CLEAN_PATH}{filename}", rate, reduced_noise)

denoise()