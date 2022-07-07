## Text translation using Transformers
## The core idea behind a transformer model is self-attention—the ability to attend
## to different positions of the input sequence to compute a representation of that sequence.

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