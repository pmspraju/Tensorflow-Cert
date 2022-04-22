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

plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens)
plt.title('Token IDs')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens != 0)
plt.title('Mask')
plt.show()
###############################################################################
