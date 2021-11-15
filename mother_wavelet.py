from scipy.stats import entropy
from scipy.signal import normalize
import numpy as np

s = [0.5,0.5]

print(entropy(s, base=2))

signal = [1,2,3,4]
signal = np.divide(signal,sum(signal))
#signal = (signal - np.mean(signal)) / np.std(signal)

print(signal)
print(sum(signal))