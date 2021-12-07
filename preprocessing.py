import numpy as np
from scipy import signal
import scipy.io.wavfile as wave
import matplotlib.pyplot as plt

class process:
    # Methods for preprocessing audio field recordings

    def __init__(self, directory):
        self.sampleRate, self.recording = self.load(directory)


    @staticmethod
    def load(directory):
        # Load recording
        sampleRate, recording = wave.read(f'{directory}.wav')
        return sampleRate, recording


    def normalise(self, directory, write=False):
        # Normalise signal
        self.recording = (self.recording - np.mean(self.recording)) / np.std(self.recording)

        if write:
            wave.write(f'{directory}.wav', self.sampleRate, self.recording)

        print('Audio normalised')


    def downSample(self, rate=16000):
        # Down sample audio files
        if self.sampleRate != rate:
            N = round((len(self.recording) * rate)/ self.sampleRate)
            self.recording = signal.resample(self.recording, N)
            self.sampleRate = rate
            print(f'Audio downsampled to {rate}')


    def generateRef(self, directory, downSample=False):
        # Save reference spectrogram as .npy file
        if downSample:
            self.downSample()

        fr, tr, Sr = signal.spectrogram(self.recording, fs=self.sampleRate)

        np.save(f'{directory}.npy', Sr)
        print('Reference saved')


    def mono(self):
        # Check and convert to mono-channel
        try:
            if self.recording.shape[1] > 1:
                print('Converted to mono-channel')
                return self.recording[:,0]
        except:
            return self.recording


# directory = './recordings/original/double_channel'
# r = process(directory)