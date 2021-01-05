# This is a sample Python script.
import sys
import tensorflow as tf
import sklearn
import numpy as np
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from methods import custom_standardization
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print(sys.version)
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    print('Tensorflow version is {}.'.format(tf.__version__))
    print('The Pandas version is {}.'.format(pd.__version__))
    print('The Numpy version is {}.'.format(np.__version__))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    max_features = 10
    sequence_length = 4
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)
    examples = [
        "The movie was great!",
        "The movie was okay.",
        "The movie was terrible..."
    ]
    vectorize_layer.adapt(examples)
    text = tf.expand_dims(examples[1], -1)
    ttl = vectorize_layer(text)
    examples_seq = []
    for i in range(len(examples)):
        text = tf.expand_dims(examples[i], -1)
        examples_seq.append(vectorize_layer(text))
    print(examples_seq)

