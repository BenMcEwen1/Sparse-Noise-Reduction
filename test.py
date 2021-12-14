import matplotlib.pyplot as plt
import pywt
import time
import numpy as np

s = [0]
level = 10

for l in range(level):
    s.append(s[-1] + 2**l)

# plt.plot(s)

c = []

signal = np.ones(100000)

for l in range(1,level):
    t1 = time.time()
    coeffs = pywt.wavedec(signal, wavelet='dmey',mode='symmetric', level=l)
    print(f"Length: {len(coeffs[0])}")
    t2 = time.time()

    diff = (t2 - t1) / 2**l
    c.append(diff)

plt.plot(c)
plt.show()