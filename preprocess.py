from pipeline import Pipeline 
import pywt
import scipy.io.wavfile as wave
from scipy.stats import kstest, uniform
import matplotlib.pyplot as plt
import numpy as np
import random


class Process:

  def __init__(self, directory, level=5, wavelet='dmey', threshold=12): 
    sampleRate, signal = wave.read(directory)
    signal = signal[0:100000] # Temporary so that they are the same length when compared
    self.sampleRate = sampleRate
    self.signal = signal
    self.level = level
    self.wavelet = wavelet
    self.threshold = threshold
    self.length = len(signal)
    en = self.entropy(signal)
    var = self.variance(signal)
    print(f"Signal entropy: {en}")
    print(f"Signal variance: {var}")

  def entropy(self, data):
    # Measure complexity of data packets
    #data = data[0:3000] # Test to ensure packet length is the same
    norm = np.linalg.norm(data)
    e = data[np.nonzero(data)]**2 * np.log(data[np.nonzero(data)]**2)
    return np.sum(e)

  def variance(self, data):
    # Measure the variance of data packets
    #data = data[0:3000] # Test to ensure packet length is the same
    norm = np.linalg.norm(data)
    return np.var(data/norm)

  def KS(self, data):
    # Kolmogororov-Smirnov test of uniformity
    u = kstest(data, uniform(loc=0.0, scale=len(data)).cdf)
    return u

  def randomness(self, N=500):
    # Random signal for reference
    rand = np.random.uniform(-1,1,size=N)
    uni = self.KS(rand)

    print(f'Uniformity of randomness: {uni}')

    plt.figure(1)
    plt.plot(rand)
    plt.show()

  def denoise(self, plot=False):
    # WPD with removal of noisy packets
    coeffs = pywt.wavedec(self.signal, self.wavelet, level=self.level)

    if plot:
      self.decomposition()

    for i,coeff in enumerate(coeffs):
      en = self.entropy(coeff)
      var = self.variance(coeff)
      uni = self.KS(coeff)

      print('---')
      print(f'Entropy: {en}')
      print(f'Variance: {var}')
      print(f'Uniformity: {uni}')

      if en > self.threshold: 
          coeffs[i] = np.zeros_like(coeffs[i])

    return coeffs

  def reconstruct(self, coeffs):
    # Inverse WPD to reconstruct signal
    signal = pywt.waverec(coeffs, self.wavelet)
    return signal

  def decomposition(self):
    # Plot decomposition packets
    fig, axarr = plt.subplots(nrows=self.level, ncols=2, figsize=(6,6))

    data = self.signal

    for ii in range(self.level):
        (data, coeff_d) = pywt.dwt(data, self.wavelet)
        axarr[ii, 0].plot(data, 'r')
        axarr[ii, 1].plot(coeff_d, 'g')
        axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
        axarr[ii, 0].set_yticklabels([])

        if ii == 0:
            axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
            axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
        axarr[ii, 1].set_yticklabels([])
    plt.tight_layout()
    plt.show()

  def spectrogram(self, signal, plot=False):
    # Return spectrogram of data packet
    spectrum, frequencies, time, axis = plt.specgram(signal, Fs=self.sampleRate)

    if plot == True:
      plt.xlabel('Time (s)')
      plt.ylabel('Frequency (Hz)')
      plt.show()

    return spectrum, frequencies, time, axis


recording = Process('./recordings/cat.wav')
s = recording.signal

recording.randomness()

spectrum, frequencies, time, axis = recording.spectrogram(s, True)

spectrum = np.array(spectrum)

coeffs = recording.denoise(True)
s2 = recording.reconstruct(coeffs)
recording.spectrogram(s2, True)

# -------------------------
# mag = []
# for c in spectrum.T:
#   mag.append(sum(c)/len(c))

# plt.plot(mag)
# plt.show()
