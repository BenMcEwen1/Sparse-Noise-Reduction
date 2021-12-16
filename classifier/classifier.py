import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import pandas as pd

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(1234)


# Import dataset and convert to dataset object
df = pd.read_csv('classifier/dataset.csv')

def get_dataset(df):
    filename_ds = tf.data.Dataset.from_tensor_slices(df.filename)
    label_ds = tf.data.Dataset.from_tensor_slices(df.label)
    return tf.data.Dataset.zip((filename_ds, label_ds))

dataset = get_dataset(df)
dataset.element_spec


# Prepare dataset
dirpath = 'classifier/audio/'

def loadAudio(filename, label):
    audio = tf.io.read_file(dirpath + filename)
    audio, sampleRate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16000)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    return audio, label

def prepare(dataset, shuffle_buffer_size=2, batch_size=2):
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(loadAudio, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

dataset = prepare(dataset)
dataset.element_spec


# Load Yamnet model (Transfer learning)
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_embedding(audio, label):
  # Extract yamnet embedding 
  scores, embeddings, spectrogram = yamnet_model(audio)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings, tf.repeat(label, num_embeddings))

dataset = dataset.map(extract_embedding).unbatch()
dataset.element_spec



def split(dataset):
    # Shuffle and split dataset
    dataset = dataset.shuffle(300) # Shuffle
    
    test = dataset.take(50) 
    remaining = dataset.skip(50)
    train = remaining.take(150)
    val = remaining.skip(100)

    test = test.cache().batch(5).prefetch(AUTOTUNE)
    train = train.cache().batch(5).prefetch(AUTOTUNE)
    val = val.cache().batch(5).prefetch(AUTOTUNE)

    return test, train, val

test, train, val = split(dataset)


def createModel():
    # Create model
    classes = {'possum': 1, 'not possum': 0}

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(classes))
    ], name='my_model')

    model.summary()

    return model, classes

model, classes = createModel()


# Compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam", metrics=['accuracy'])


# Train model
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

model.fit(train, validation_data=val, epochs=10, callbacks=callback)

# Evaluate model
loss, accuracy = model.evaluate(test)
print("Loss: ", loss)
print("Accuracy: ", accuracy)


# Test 
audio,_ = loadAudio('p3.wav', 1)
scores, embeddings, spectrogram = yamnet_model(audio)
result = model(embeddings).numpy()

inferred_class = list(classes.keys())[result.mean(axis=0).argmax()]
print(f'The main sound is: {inferred_class}')


# Combine and save model (Replace path)
class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)

saved_model_path = '/home/cosc/student/bmc142/dataset/classifier/' # Replace

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer('https://tfhub.dev/google/yamnet/1',trainable=False, name='yamnet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)

serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)