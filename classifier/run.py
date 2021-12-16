import tensorflow as tf

reloaded_model = tf.saved_model.load('classifier/')

audio = tf.io.read_file('/home/cosc/student/bmc142/dataset/classifier/audio/c1.wav')
audio, sampleRate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16000)

classes = {'possum': 1, 'not possum': 0}
result = reloaded_model(audio)
classification = my_classes[tf.argmax(result)]
print(f'The main sound is: {classification}')