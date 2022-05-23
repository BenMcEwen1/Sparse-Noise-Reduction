import tensorflow as tf

def loadAudio(filename, label):
    dirpath = 'classifier/audio/'

    audio = tf.io.read_file(dirpath + filename)
    audio, sampleRate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16000)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    return audio, label


# Test 
classes = {'possum': 1, 'unkown': 0}
audio,_ = loadAudio('possum16k.wav', 1)

model = tf.keras.models.load_model('./classifier/model')
result = model(audio).numpy()

print(result)

inferred_class = list(classes.keys())[result.mean(axis=0).argmax()]
print(f'The main sound is: {inferred_class}')