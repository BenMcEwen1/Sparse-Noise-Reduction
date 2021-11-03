import scipy.io.wavfile as wave
import numpy as np
import pywt
from scipy.signal import chirp
import matplotlib.pyplot as plt


# Chirp (Test signal)
t = np.linspace(0, 10, 30000)
signal = chirp(t, f0=0.1, f1=300, t1=10, method='linear')
noise = np.random.rand(30000) * 0.1
signal += noise

# Generate Tree
wp = pywt.WaveletPacket(signal, wavelet='dmey', mode='symmetric', maxlevel=3)

# Show mother wavelet families
print(pywt.families())

# Spectrogram of raw signal
plt.figure(1)
plt.specgram(signal, Fs=3000)
plt.show()

# Remove packets then reconstruct
remove = ['d','da','dd','add ','dda','ddd','dad','daa']

for level in range(1,wp.maxlevel+1):
    for node in wp.get_level(level, 'freq'):
        if (node.path in remove):
            print(f"{node.path} deleted")
            # del wp[node.path]
            wp[node.path].data = np.zeros(len(node.data))
            # wp[node.path].data = node.data * 0.5

            # Thresholding
            #wp[node.path].data = pywt.threshold(node.data,np.mean(abs(node.data)),'soft')
            wp.reconstruct(True)
        else:
            print(node.path)


# Spectrogram of denoised reconstructed signal
plt.figure(2)
plt.specgram(wp.data, Fs=3000)
plt.show()

# Approximate, detailed and reconstructed signal (Weird stretching)
(A, D) = pywt.dwt(wp.data, 'dmey')
d = pywt.idwt(A, None, 'dmey')
plt.figure(3)
plt.plot(A)
plt.plot(D)
plt.plot(d)
plt.show()

#------------------------------------------------------------

a = wp['a'].data #First Node
d = wp['d'].data #Second Node
#The second floor
aa = wp['aa'].data 
ad = wp['ad'].data 
dd = wp['dd'].data 
da = wp['da'].data 
#Layer 3
aaa = wp['aaa'].data 
aad = wp['aad'].data 
ada = wp['add'].data 
add = wp['ada'].data 
daa = wp['dda'].data 
dad = wp['ddd'].data 
dda = wp['dad'].data 
ddd = wp['daa'].data

plt.figure(3, figsize=(15, 10))
plt.subplot(4,1,1)
plt.plot(signal)
#First floor
plt.subplot(4,2,3)
plt.plot(a)
plt.subplot(4,2,4)
plt.plot(d)
#The second floor
plt.subplot(4,4,9)
plt.plot(aa)
plt.subplot(4,4,10)
plt.plot(ad)
plt.subplot(4,4,11)
plt.plot(dd)
plt.subplot(4,4,12)
plt.plot(da)
#Layer 3
plt.subplot(4,8,25)
plt.plot(aaa)
plt.subplot(4,8,26)
plt.plot(aad)
plt.subplot(4,8,27)
plt.plot(add)
plt.subplot(4,8,28)
plt.plot(ada)
plt.subplot(4,8,29)
plt.plot(dda)
plt.subplot(4,8,30)
plt.plot(ddd)
plt.subplot(4,8,31)
plt.plot(dad)
plt.subplot(4,8,32)
plt.plot(daa)
plt.show()