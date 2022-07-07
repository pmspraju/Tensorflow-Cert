# Subword tokenizer
# This implements BertTokenizer class is a higher level interface.
# It includes BERT's token splitting algorithm and a WordPieceTokenizer.
# It takes sentences as input and returns token-IDs.

import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# Load the dataset portuguese to english translation dataset
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

# separate out english and portuguese
train_en = train_examples.map(lambda pt, en:en)
train_pt = train_examples.map(lambda pt, en:pt)

# We use bert tokenizer
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

# Set the parameters
bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = 8000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

# Create the vocabulary for portuguese sentences
pt_vocab = bert_vocab.bert_vocab_from_dataset(
    train_pt.batch(1000).prefetch(2),
    **bert_vocab_args
)

print(pt_vocab[:10])
print(pt_vocab[100:110])

# Write the vocabulary to a file. This file is used to create the token-ids
filepath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\transformer'

def write_vocab_file(filepath, vocab):
  with open(filepath, 'w', encoding="utf-8") as f:
    for token in vocab:
      print(token, file=f)

pt_filepath = os.path.join(filepath, 'pt_vocab.txt')
write_vocab_file(pt_filepath, pt_vocab)

# Generate vocabulary for english sentences
en_vocab = bert_vocab.bert_vocab_from_dataset(
    train_en.batch(1000).prefetch(2),
    **bert_vocab_args
)
en_filepath = os.path.join(filepath, 'en_vocab.txt')
write_vocab_file(en_filepath, en_vocab)

# Once we have the vocabulary, build the tokenizer
pt_tokenizer = text.BertTokenizer(pt_filepath, **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer(en_filepath, **bert_tokenizer_params)

# Tokenize the examples -> (batch, word, word-piece)
token_batch = en_tokenizer.tokenize(engSen)
# Merge the word and word-piece axes -> (batch, tokens)
token_batch = token_batch.merge_dims(-2,-1)

for ex in token_batch.to_list():
  print(ex)

# Detokenize - to get the words back from the indices
# method 1 - use tf.gather
# Lookup each token id in the vocabulary.
txt_tokens = tf.gather(en_vocab, token_batch)
# Join with spaces.
print(tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1))

# To re-assemble words from the extracted tokens, use the BertTokenizer.detokenize method
words = en_tokenizer.detokenize(token_batch)
print(tf.strings.reduce_join(words, separator=' ', axis=-1))

# Add the start and end tokens as delimiters to detokenized sentence
START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

def add_start_end(ragged):
  count = ragged.bounding_shape()[0]
  starts = tf.fill([count,1], START)
  ends = tf.fill([count,1], END)
  return tf.concat([starts, ragged, ends], axis=1)

words = en_tokenizer.detokenize(add_start_end(token_batch))
tf.strings.reduce_join(words, separator=' ', axis=-1)

# 1. They want to generate clean text output, so drop reserved tokens like [START], [END] and [PAD].
# 2. They're interested in complete strings, so apply a string join along the words axis of the result.
def cleanup_text(reserved_tokens, token_txt):
  # Drop the reserved tokens, except for "[UNK]".
  bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
  bad_token_re = "|".join(bad_tokens)

  bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
  result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

  # Join them into strings.
  result = tf.strings.reduce_join(result, separator=' ', axis=-1)

  return result

# Testing the cleaning up
token_batch = en_tokenizer.tokenize(engSen).merge_dims(-2,-1)
words = en_tokenizer.detokenize(token_batch)
print(words)
print(cleanup_text(reserved_tokens, words).numpy())

# Export the model
class CustomTokenizer (tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.vocab = tf.Variable(vocab)

        ## Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()


    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

tokenizers = tf.Module()
tokenizers.pt = CustomTokenizer(reserved_tokens, pt_filepath)
tokenizers.en = CustomTokenizer(reserved_tokens, en_filepath)

# Export the model
model_name = 'ted_hrlr_translate_pt_en_converter'
modelpath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\transformer'
modelpath = os.path.join(modelpath, modelpath)
#tf.saved_model.save(tokenizers, modelpath)

# Re-load the saved model
reloaded_tokenizers = tf.saved_model.load(modelpath)
reloaded_tokenizers.en.get_vocab_size().numpy()

# Test
tokens = reloaded_tokenizers.en.tokenize(['Hello TensorFlow!'])
print(tokens.numpy())

text_tokens = reloaded_tokenizers.en.lookup(tokens)
print(text_tokens)

round_trip = reloaded_tokenizers.en.detokenize(tokens)
print(round_trip.numpy()[0].decode('utf-8'))








