# Import the libraries.
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers

import numpy as np
import os
import time
import re
import string

print("----TensorFlow version:", tf.__version__)
print("----Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Download the dataset if not already
path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data'
loadPath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\Models\2'
chkptpath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\Checkpoints'

folder = 'nlp'
abs_path = os.path.join(path, folder)
abs_path = os.path.join(abs_path, 'shakespeare')

# Read the file
filepath = os.path.join(abs_path, 'essay.txt')
with open(filepath,'rb') as f:
    text = f.read().decode(encoding='utf-8') #use utf-8 to deocde special characters

print('----Length of text: {} characters'.format(len(text)))

# join the text, at next line, in to a list
corpus = text.lower().split("\n")

# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))
print(vocab)

# Use string lookup to map each different vocab to a number id
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab))

# Inverse mapping of tokens to characters
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True)

# Tokenize by using string look up layer
tokens = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
#print(tokens.numpy())

# Create a tf dataset from the tokens
token_dataset = tf.data.Dataset.from_tensor_slices(tokens)

# create window sequences for Rnn
n_steps = 25
window_size = n_steps + 1

# Nested dataset, dataset of datasets, with window_size.
dataset_train = token_dataset.window(window_size, shift=1, drop_remainder = True)

# # print the shape of the dataset
# for item in dataset_train.take(1):
#     print([i.numpy() for i in item])
#     break

# change each window dataset to a batch and flat the list
dataset_train = dataset_train.flat_map(lambda window: window.batch(window_size))

# # print the shape
# for item in dataset_train:
#     print(item)
#     break

# Convert the tokens back to strings
# Reduce the group of tokens to group of characters (words)
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# Create the sequence and target.
# Target is sequence created by one-char shift of original sequence.
# Ex - Sequence - Hello world; Target- ello worldx
dataset_train = dataset_train.map(lambda window: (window[:-1], window[1:]))

for input,target in dataset_train.take(1):
    print("----Input sequence:", text_from_ids(input).numpy())
    print("----Target sequence:", text_from_ids(target).numpy())

# Create the batches for training. Use prefetch for performance
BATCH_SIZE = 32
BUFFER_SIZE = 10000
dataset_batch = (
    dataset_train
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
#    .cache()
    .prefetch(tf.data.experimental.AUTOTUNE))

# Hyper parameters
# Build the model
vocab_size = len(ids_from_chars.get_vocabulary())
print("Vocabulary size:", vocab_size)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
embedding_dim = 25
rnn_units = 1024
EPOCHS = 100

# Create the RNN layers
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=n_steps))
# model.add(tf.keras.layers.GRU(rnn_units, return_sequences=True))
# model.add(tf.keras.layers.Dense(vocab_size))
# model.summary()

# Check points call back
# Directory where the checkpoints will be saved
checkpoint_dir = os.path.join(chkptpath, '3')
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Compile and fit the model
mpath = os.path.join(loadPath,'rnn_1.h5')
# model.compile(optimizer='adam', loss=loss)
# history = model.fit(dataset_batch, epochs=EPOCHS, callbacks=[checkpoint_callback])
# model.save(mpath)

# Load the saved model
load_model = tf.keras.models.load_model(mpath)
load_model.summary()

def prediction(seed_seq, number_of_char):
    result = seed_seq
    for i in range(number_of_char):
        test_seq = tf.constant([seed_seq])  # length = 19
        input_chars = tf.strings.unicode_split(test_seq, 'UTF-8')
        input_ids = ids_from_chars(input_chars).to_tensor()

        predict_seq = load_model.predict(input_ids)
        # Predicted sequence will be of shape (batch, length of test sequence, length of vocabulary size) = (1,19,61)
        #print(predict_seq)

        # We use prediction of last character in the test sequence
        predict_seq = predict_seq[:,-1,:]
        #print(predict_seq)

        # Select the logit with hight probability
        predicted_ids = tf.random.categorical(predict_seq, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        #print(predicted_ids)

        # Convert from token ids to characters
        predicted_chars = chars_from_ids(predicted_ids)
        predicted_chars = "".join([b.decode("utf-8") for b in predicted_chars.numpy()])

        result = result + predicted_chars
        seed_seq = result

    return result

seed_seq = 'During the past cen'
pred = prediction(seed_seq, 50)

print("----Test sequence:",seed_seq)
print("----Predicted character:", pred)

# Another model
# model_1 = tf.keras.models.Sequential()
# model_1.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=n_steps))
# model_1.add(tf.keras.layers.GRU(rnn_units, return_sequences=True))
# model_1.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True)))
# model_1.add(tf.keras.layers.LSTM(100, return_sequences=True))
# model_1.add(tf.keras.layers.Dense(vocab_size))
# model_1.summary()
#
# # Compile and fit the model
m1path = os.path.join(loadPath,'rnn_2.h5')
# model_1.compile(optimizer='adam', loss=loss)
# history = model_1.fit(dataset_batch, epochs=EPOCHS)
# model_1.save(m1path)

# Load the saved model
load_model = tf.keras.models.load_model(m1path)
load_model.summary()

seed_seq = 'During the past cen'
pred = prediction(seed_seq, 50)

print("----Test sequence:",seed_seq)
print("----Predicted character:", pred)