from pipeline import Pipeline 
import pywt
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt
import numpy as np


class Process:

  def __init__(self, directory, level=5, wavelet='sym5', threshold=12):
    sampleRate, signal = wave.read(directory)
    self.sampleRate = sampleRate
    self.signal = signal
    self.level = level
    self.wavelet = wavelet
    self.threshold = threshold

  def entropy(self, data):
    # Measure randomness of data packets
    E = data**2/len(data)
    P = E/sum(E)
    S = -sum(P*np.log2(P))
    return S

  def noiseReduction(self):
    # WPD with removal of noisy packets
    coeffs = pywt.wavedec(self.signal, self.wavelet, level=self.level)
    
    for i,coeff in enumerate(coeffs):
      en = self.entropy(coeff)
      print(en)
      if en > self.threshold:
          coeffs[i] = np.zeros_like(coeffs[i])

    return coeffs

  def reconstruct(self, coeffs):
    # Inverse WPD to reconstruct signal
    signal = pywt.waverec(coeffs, self.wavelet)
    return signal

  def spectrogram(self, signal, plot=False):
    # Return spectrogram of data packet
    spectrum, frequencies, time, axis = plt.specgram(signal, Fs=self.sampleRate)

    if plot == True:
      plt.show()

    return spectrum, frequencies, time, axis


recording = Process('./recordings/miaow_16k.wav')
s = recording.signal
spectrum, frequencies, time, axis = recording.spectrogram(s, True)

plt.pcolormesh(time, frequencies, spectrum)
plt.show()