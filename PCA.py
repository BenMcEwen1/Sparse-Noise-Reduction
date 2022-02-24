from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import librosa
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio


recording = './recordings/downsampled/stoat16k.wav'
audio, sampleRate = librosa.load(recording, sr=None)

plt.specgram(audio, Fs=sampleRate)
plt.show()

# Compute features
features = librosa.feature.mfcc(audio, sr=sampleRate)
print(features.shape)

# Scale the features
features = features.astype(float)
features = scale(features)

# Create and apply model
model = PCA(n_components=10, whiten=True)
model.fit(features.T)
compressed = model.transform(features.T)
print(compressed.shape)

# Plot number of principle components returned
# print(model.components_.shape)
# plt.scatter(compressed[:,0], compressed[:,1])
# plt.show()

reconstruct = model.inverse_transform(compressed)
plt.imshow(reconstruct)
plt.show()