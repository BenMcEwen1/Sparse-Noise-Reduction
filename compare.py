import numpy as np
from scipy import signal

cat = np.load('./reference/cat.npy')
possum = np.load('./reference/possum.npy')

print(cat.shape)
print(possum.shape)

print(signal.correlate(cat,possum))

