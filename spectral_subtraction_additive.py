from scipy.io import wavfile
import noisereduce as nr
import os, random
import librosa
import numpy as np

# BASE = "./field audio/"
# NOISY_PATH = './noise segments/'
# CLEAN_PATH = './dataset/train/clean/'
# NOISE_PATH = './noise segments/'

def baseline(path):
    rms = []
    for _,_,filenames in os.walk(path):
        for filename in sorted(filenames):
            signal,_ = librosa.load(f'{path}{filename}', sr=16000)
            mean_rms = np.mean(librosa.feature.rms(y=signal))
            rms.append(mean_rms)
    print(f"Minimum Average RMS {20*np.log10(np.min(rms))}")
    print(f"Maximum Average RMS {20*np.log10(np.max(rms))}")
    return 20*np.log10(np.mean(rms))


def sample_noise(target, noise_path='./noise segments/'):
    filename = random.choice(os.listdir(noise_path))
    signal,_ = librosa.load(f'{noise_path}{filename}', sr=16000)

    # Scale noise segment to reach target RMS
    current_rms = np.mean(librosa.feature.rms(y=signal))
    scale_factor = (target/current_rms)
    print(f'RMS factor: {scale_factor}')
    print(f'target (dB): {20*np.log10(target)}')

    scaled_signal = np.multiply(signal, scale_factor)
    scaled_signal = np.repeat(scaled_signal, 2)
    return scaled_signal


def additive_noise(path, rms_difference=5):
    baseline_rms = baseline(path)
    print(f'Dataset mean RMS: {baseline_rms}')
    target = 10**((baseline_rms - rms_difference)/20)

    for _,_,filenames in os.walk(path):
        for filename in filenames:
            if filename not in ['cat.wav', 'PossumNoisy.wav']:
                scaled = sample_noise(target) # Randomly selected noise sample
                clean,rate = librosa.load(f'{path}{filename}', sr=16000)
                combined = np.add(clean, scaled)
                wavfile.write(f"./rms/rms{rms_difference}/original/{filename}", rate, combined)

path = './audio/predator/'
additive_noise(path, rms_difference=5)