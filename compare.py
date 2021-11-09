import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
from scipy.fft import fftshift
import matplotlib.pyplot as plt

sampleRate, possum = wave.read('recordings/possum_clean.wav')
sampleRate, ref = wave.read('recordings/clean_snip.wav')


# 1-d correlation of time-domain signal
possum = possum/max(possum)
ref = ref/max(ref)

cor = signal.correlate(possum,ref)

plt.figure(1)
plt.plot(cor)
plt.show()

# # 2-d correlation of spectrograms
# sp, fp, tp, imp = plt.specgram(possum, sampleRate)
# sr, fr, tr, imr = plt.specgram(ref, sampleRate)

# # plt.pcolormesh(tp, fp, Sp)
# # plt.ylabel('Frequency [Hz]')
# # plt.xlabel('Time [sec]')
# # plt.show()

# corr = signal.correlate2d(sp, sr, boundary='symm', mode='same')

# plt.imshow(sp)
# # plt.imshow(corr)
# plt.show()
