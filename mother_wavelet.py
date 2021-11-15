from scipy.stats import entropy, kstest, uniform
from scipy.signal import normalize
import numpy as np
import random


# Test entropy
s = [0.5,0.5]

print(entropy(s, base=2))

signal = [1,2,3,4]



# Test Uniformity

def KS(data):
    # Kolmogororov-Smirnov test of uniformity
    Uniformity = kstest(data, uniform(loc=0.0, scale=len(data)).cdf)
    return Uniformity.statistic

signal = [1,1,1,1,2,3,4,3,2,1,1,1,1,1,1]
signal = np.divide(signal,sum(signal))

print(KS(signal))
