from scipy.stats import entropy, kstest, uniform
from scipy.signal import normalize
import numpy as np
import random
import scipy.io.wavfile as wave
from scipy import signal
import matplotlib.pyplot as plt

filename = 'denoised/possum44k'
sampleRate, s = wave.read(f'recordings/{filename}.wav')

fp, tp, Sp = signal.spectrogram(s, fs=sampleRate)

e = 1e-11
Sp = np.log(Sp + e)

possum = np.load('reference/normalised/possum.npy')

# Test
test = np.ones((129,31)) * -1

test[5:15] = 10

testSignal = np.ones((129,1719))

testSignal[5:15] = 10
testSignal.T[0:50] = 1
testSignal.T[500:1400] = 1
testSignal.T[1600:1719] = 1

print(test.shape)
print(testSignal.shape)


# test = np.flipud(test)

# plt.imshow(test)
# plt.show()

# plt.imshow(testSignal)
# plt.show()



# # Absolute ref
# possum = abs(possum)
# plt.imshow(possum)
# plt.show()

# # Normalise ref
# possum = possum / possum.max()
# plt.imshow(possum)
# plt.show()

# # Square
# possum = possum ** 2

#-------------------------------------------

# Absolute signal
# Sp = abs(Sp)
# plt.imshow(Sp)
# plt.show()

# Normalise signal
# Sp = Sp / Sp.max()
# plt.imshow(Sp)
# plt.show()


c = signal.correlate(Sp, possum, 'valid')

# c = signal.convolve2d(testSignal, test, mode="valid", boundary="wrap")
print(c)
# c = abs(np.subtract(c[0],max(c[0])))  
c = c[0]
# c = np.interp(c, (c.min(),c.max()), (0,1)) 

print(c.shape)
plt.plot(c)
plt.show()

