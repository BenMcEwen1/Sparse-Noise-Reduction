import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
import matplotlib.pyplot as plt


# Chirp (Test signal)
t = np.linspace(0, 10, 15000)
signal = chirp(t, f0=0.1, f1=300, t1=10, method='linear')
noise = np.random.rand(15000) * 0.1
signal += noise


wp = pywt.WaveletPacket(signal, wavelet='dmey', mode='symmetric', maxlevel=3)


decomp = pywt.swt(signal, 'dmey', level=1)

print((decomp))



# approx = []
# detail = []

# coeffs = [signal]


# for level in range(1, wp.maxlevel+1):
#     print(f'Level: {level}')
#     levels = []
#     for coeff in coeffs:
#         (A,D) = pywt.dwt(coeff, wavelet='dmey', mode='symmetric', axis=-1)
#         levels.append(A)
#         levels.append(D)
#     # detail.append(A)
#     # approx.append(D)

#     coeffs.append(levels)
    

# print(f'Number of nodes: {len(coeffs)}')

