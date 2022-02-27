import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from methods import custom_standardization, RnnModel, OneStep, CustomTraining

import numpy as np
import os
import time
import re
import string

print(tf.__version__)

# Download the dataset if not already
path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data'
folder = 'nlp'
abs_path = os.path.join(path, folder)
abs_path = os.path.join(abs_path, 'shakespeare')
if not os.path.exists(os.path.join(abs_path, 'shakespeare.txt')):
    sp_txt = tf.keras.utils.get_file('shakespeare.txt',
                                     cache_subdir=abs_path,
                                     origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt',
                                     )
    sp_dir = abs_path
else:
    sp_dir = abs_path

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

file = os.path.join(sp_dir, 'shakespeare.txt')
text = open(file, 'rb').read().decode(encoding='utf-8')
print('Length of text: {} characters'.format(len(text)))
print(text[:250])

corpus = text.lower().split("\n")

# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))
print(vocab)

# Vectorize layer consider words, but in this example we are using character level prediction.
max_features = 30000
sequence_length = 250

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(np.asarray(corpus))


def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)


# Only dataset.batch can be used in this way. Or we need to create a model for vectorization layer and use model.add
# and model.predcit to get the output.
# po = [vectorize_text(tf.constant(x)) for x in corpus] This doesnt work.
# ds = tf.data.Dataset.from_tensor_slices(corpus)
# ds = ds.batch(2).map(vectorize_text)

# Use string lookup to map each different vocab to a number id
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab))

# Inverse mapping of tokens to characters
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True)


# Reduce the group of tokens to group of characters (words)
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


tokens = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
# print(tokens.numpy())

# Create a dataset from the tokens
token_dataset = tf.data.Dataset.from_tensor_slices(tokens)

# Print characters of the sample ids
# for ids in token_dataset.take(10):
#     print(chars_from_ids(ids).numpy().decode('utf-8'))

# Create the sequences of tokens of fixed length
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)
sequences = token_dataset.batch(seq_length + 1, drop_remainder=True)

# Print the words (group of characters) for each sequence of tokens
for seq in sequences.take(5):
    print(text_from_ids(seq).numpy())


# As we are planning to predict the next character
# We create the target sequence as below
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

# Print a sample input and target
# for input_example, target_example in dataset.take(1):
#     print("Input :", text_from_ids(input_example).numpy())
#     print("Target:", text_from_ids(target_example).numpy())

# Create batches of the input using prefetch.
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

print(dataset)

# Build the model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

model = RnnModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    # We can see the output shape has vocab_size, because the output dense layer has
    # neurons equal to vocab_size

model.summary()
# print(example_batch_predictions[0])

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# print(sampled_indices)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

# Print the input sample sequence and its prediction
print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

# Add loss and Optimizer
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)
tf.exp(mean_loss).numpy()

# Compile the model
model.compile(optimizer='adam', loss=loss)

# Check points call back
# Directory where the checkpoints will be saved
chkptpath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\Checkpoints'
checkpoint_dir = os.path.join(chkptpath, '2')
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 1

#history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Predict one step. Create the model
#one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

# run the one step model in a loop
start = time.time()
states = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]

#for n in range(1000):
#  next_char, states = one_step_model.generate_one_step(next_char, states=states)
#  result.append(next_char)

result = tf.strings.join(result)
end = time.time()

print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)

print(f"\nRun time: {end - start}")

model_c = CustomTraining(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

model_c.compile(optimizer = tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# model_c.fit(dataset, epochs=1)

# Custom training loop
EPOCHS = 10

mean = tf.metrics.Mean()

for epoch in range(EPOCHS):
    start = time.time()

    mean.reset_states()
    for (batch_n, (inp, target)) in enumerate(dataset):
        logs = model_c.train_step([inp, target])
        mean.update_state(logs['loss'])

        if batch_n % 50 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch + 1, batch_n, logs['loss']))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model_c.save_weights(checkpoint_prefix.format(epoch=epoch))

    print()
    print('Epoch {} Loss: {:.4f}'.format(epoch + 1, mean.result().numpy()))
    print('Time taken for 1 epoch {} sec'.format(time.time() - start))
    print("_"*80)

model_c.save_weights(checkpoint_prefix.format(epoch=epoch))

# Predict one step. Create the model
# run the one step model in a loop
start = time.time()
states = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]

# Predict one step. Create the model
one_step_model = OneStep(model_c, chars_from_ids, ids_from_chars)

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()

print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)

print(f"\nRun time: {end - start}")