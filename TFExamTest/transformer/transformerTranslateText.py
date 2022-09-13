## Text translation using Transformers
## The core idea behind a transformer model is self-attention—the ability to attend
## to different positions of the input sequence to compute a representation of that sequence.

import os
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# Load the dataset from tfds
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# See example input portuguese sentences
portguese = []; english = [];
for portSen, engSen in train_examples.batch(3).take(1):
    for port in portSen.numpy():
        portguese.append(port.decode('utf-8'))

    for sen in engSen.numpy():
        english.append(sen)

    for i,j in zip(portguese,english):
        print('Portguese  Sentence:', i)
        print('English Translation:', j)
        print()

# Tokenize and Detokenize. Use saved model from subwordTokenizer.py
model_name = 'saved_model.pb'
modelpath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\transformer'
modelpath = os.path.join(modelpath, modelpath)
tokenizers = tf.saved_model.load(modelpath)

print([item for item in dir(tokenizers.en) if not item.startswith('_')])
print()

# Tokenize
encoded = tokenizers.en.tokenize(engSen)

# Detokenize the tokens to see how it goes
round_trip = tokenizers.en.detokenize(encoded)


for i, j, k in zip(english, encoded, round_trip.numpy()):
    print('English  Sentence:', i)
    print('Encoded tokens:', j)
    print('Detokenized sentence:', k.decode('utf-8'))
    print()


