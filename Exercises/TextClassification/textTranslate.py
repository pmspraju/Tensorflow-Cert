############################################
# Text Translation using attention         #
############################################
# Dataset:  http://www.manythings.org/anki/#
############################################
import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#print("----TensorFlow version:", tf.__version__)
#print("----Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Download the file
import pathlib
path_to_zip = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data\nlp'
path_to_file = pathlib.Path(path_to_zip)/'spa-eng/spa.txt'

# Load the file and print sample text
text = path_to_file.read_text(encoding = 'utf-8')
lines = text.splitlines()
pairs = [line.split('\t') for line in lines]

inp = [inp for targ, inp in pairs]
targ = [targ for targ, inp in pairs]

# print(inp[-1])
# print(targ[-1])

# Create tensorflow dataset
BUFFER_SIZE = len(inp)
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

# Print the sample from dataset
print('Number of examples: {}'.format(len(inp)))
print('Number of batches: {}'.format(len(list(dataset.as_numpy_iterator()))))

for inpBatch,targBatch in dataset.take(1):
    for input,target in zip(inpBatch,targBatch):
        print('Input spanish text:{}'.format(input))
        print('Target English Translation:{}'.format(target))
        break
    break

##################
# Pre-Processing #
###############################################################################
# First step - Text normalization -
# The standard also defines a text normalization procedure,
# called Unicode normalization, that replaces equivalent sequences of
# characters so that any two texts that are equivalent will be reduced
# to the same sequence of code points, called the normalization form or
# normal form of the original text.

# We use Normalization Form KD (NFKD) - Compatibility Decomposition
# Example
ex_string = tf.constant('¿Todavía está en casa?')
print('Raw string:{}'.format(ex_string.numpy()))
norm_string = tf_text.normalize_utf8(ex_string,'NFKD')
print('Normalized string:{}'.format(norm_string.numpy()))

# tf.strings should be used as this method will be used
# inside a tensorflow layer.
def tf_lower_and_split_punct(text):
  # Split accecented characters.
  text = tf_text.normalize_utf8(text, 'NFKD')
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

print(ex_string.numpy().decode())
print(tf_lower_and_split_punct(ex_string).numpy().decode())
###############################################################################
# Text Vectorization -
# Text Vectorization is the process of converting text into numerical representation.

# Input text processor. This include normalization and Vectorization.
max_vocab_size = 5000

input_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

# Output text processor. This include normalization and Vectorization.

output_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

# Adapt the processors on input and output
input_text_processor.adapt(inp)
output_text_processor.adapt(targ)
example_tokens = input_text_processor(inpBatch)

print('Example batch vocabulary size:{}'.format(input_text_processor.vocabulary_size()))
print('Shape of the tokens:{}'.format(tf.shape(example_tokens).numpy()))
print('Rank of the tokens:{}'.format(tf.rank(example_tokens).numpy()))

plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens)
plt.title('Token IDs')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens != 0)
plt.title('Mask')
plt.show()
###############################################################################
# Encoder architecture
# For text translation, we use encoder-decoder model with attention.
# Because, for translation, the size of the input need not be same as output
# Takes the input sequence and provies us with a context
class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.enc_units = enc_units # Number of RNN units
        self.input_vocab_size = input_vocab_size # Input vocabulary size

        # The embedding layer converts sequence of tokens to sequence of vectors
        # tokens = [1, 2, 3] -> embedded to = [[2,4,5..embedding_dim], [4,1,3..embedding_dim], [4,5,2..embedding_dim]]
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                                   embedding_dim)

        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       # Return the sequence and state
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):

        #1 tokens shape - ('batch', 's')

        # 2. The embedding layer looks up the embedding for each token.
        vectors = self.embedding(tokens) # ('batch', 's', 'embed_dim')

        # 3. The GRU processes the embedding sequence.
        #    output shape: (batch, s, enc_units)
        #    state shape: (batch, enc_units)
        output, state = self.gru(vectors, initial_state=state)

        # 4. Returns the new sequence and its state.
        return output, state , vectors

# Test the encoder class on the example input batch
# Convert the input text to tokens.

embedding_dim = 256
units = 1024

# Encode the input sequence.
encoder = Encoder(input_text_processor.vocabulary_size(),
                  embedding_dim, units)
example_enc_output, example_enc_state, embedded_vectors = encoder(example_tokens)

print(f'Input batch, shape (batch): {inpBatch.shape}')
print(f'Input batch tokens, shape (batch, s): {example_tokens.shape}')
print('Embedded tokens shape, shape (batch, s, embedded_dim):{}'.format(tf.shape(embedded_vectors).numpy()))
print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')
print(f'Encoder state, shape (batch, units): {example_enc_state.shape}')

