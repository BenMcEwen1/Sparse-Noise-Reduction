import os
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
import numpy as np


class Pipeline:

    def __init__(self, directory):
        self.directory = directory

    def extract(self):
        # Extract recordings, repalce with SD card directory
        for dirpaths, dirnames, filenames in os.walk(self.directory):
            for filename in filenames:
                print(filename)
                self.window(filename)

    def window(self, filename):
        # Split recording into windows for classification
        sampleRate, signal = wave.read(self.directory + filename)

        windowSize = sampleRate       # 1 second
        stepSize = int(sampleRate/2)  # 0.5 second

        i = 0

        while i < (len(signal) - stepSize):
            s = signal[i:i+windowSize]  # Feed this into model
            i += stepSize  

