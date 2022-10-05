#################################################
# Neural machine translation using Transformers #
#################################################

#############################
# Import relevant libraries #
#############################
import logging
import os.path
import time
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_text
import tensorflow_datasets as tfds

# Load the dataset using tensorflow datasets
# examples, metadata = tfds.load('ted_hrlr_translate/gl_to_en',
#                                with_info=True,
#                                as_supervised=True)

###############
# Set logging #
###############
logpath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data'
logfile = os.path.join(logpath,'log.txt')
logging.basicConfig(filename=logfile, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Neural Machine Translation - Transformers')
logging.info('Tensorflow version :{}'.format(tf.__version__))
logging.info(tf.config.list_physical_devices('GPU'))
logging.info("Num GPUs Available: ".format(len(tf.config.list_physical_devices('GPU'))))

#########################################
# Load and split the data from the disk #
#########################################
path_to_zip = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data\nlp'
path_to_file = pathlib.Path(path_to_zip)/'por-eng/por.txt'

# Load the file and print sample text
text = path_to_file.read_text(encoding = 'utf-8')
lines = text.splitlines()
pairs = [line.split('\t') for line in lines]

inp = [inp for targ, inp, comment in pairs]
targ = [targ for targ, inp, comment in pairs]

DATASET_SIZE= len(inp)
logging.info('Number of samples:{}'.format(DATASET_SIZE))

# Convert to tensorflow dataset
# Create tensorflow dataset
BUFFER_SIZE = len(inp)
dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)

# Set the Train, test and validation
train_size = int(0.8 * DATASET_SIZE);
val_size = int(0.1 * DATASET_SIZE)
test_size = int(0.1 * DATASET_SIZE)

logging.info('Sizes of Train:{}; Test:{}; Val:{}'.format(train_size,test_size,val_size))

# Split in to Train, validation and Test sets
train_examples = dataset.take(train_size)
test_examples = dataset.skip(train_size)
val_examples = test_examples.skip(test_size)
test_examples = test_examples.take(test_size)

# Test the dataset - print sample sentences
for pt_examples, en_examples in train_examples.batch(3).take(1):
  logging.info('> Examples in Portuguese:')
  for pt in pt_examples.numpy():
    logging.info(pt.decode('utf-8'))
  logging.info('')

  logging.info('> Examples in English:')
  for en in en_examples.numpy():
    logging.info(en.decode('utf-8'))

################
# Tokenization #
################
# Tokenize the sentences. We use pretrained Bert tokenizer
# Tokenization is the process of breaking up a sequence,
# such as a text, into tokens, for each element in that sequence.
# Commonly, these tokens are words, characters, numbers, subwords, and/or punctuation.
# The beginning of sentences and end of sentences are typically also marked by
# tokens IDs, such as '[START]' and '[END]'.

# model_name = 'ted_hrlr_translate_pt_en_converter'
# tf.keras.utils.get_file(
#     f'{model_name}.zip',
#     f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
#     cache_dir='.', cache_subdir='', extract=True
# )

# Downloaded the model at location
tokenModelPath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\transformer\tokenizer'
tokenizers = tf.saved_model.load(tokenModelPath)

logging.info([item for item in dir(tokenizers.en) if not item.startswith('_')])

# Tokenize the example sentences
encoded = tokenizers.en.tokenize(en_examples)
# De-tokenize the above tokens
round_trip = tokenizers.en.detokenize(encoded)
# Look up for token - to - words
tokens = tokenizers.en.lookup(encoded)
logging.info('> This is a padded-batch of Sentence, token IDs, Detoken sentences, Lookup:')
for sen,tok,detok,lookup in zip(en_examples.numpy(), encoded.to_list(), round_trip.numpy(), tokens.numpy()):
  logging.info('-----------')
  logging.info(sen)
  logging.info(tok)
  logging.info(detok.decode('utf-8'))
  logging.info(lookup)

# The distribution of tokens per example in the dataset is as follows:
lengths = []
for pt_examples, en_examples in train_examples.batch(1024):
  pt_tokens = tokenizers.en.tokenize(pt_examples)
  lengths.append(pt_tokens.row_lengths())

  en_tokens = tokenizers.en.tokenize(en_examples)
  lengths.append(en_tokens.row_lengths())
  print('.', end='', flush=True)

all_lengths = np.concatenate(lengths)

plt.hist(all_lengths, np.linspace(0, 500, 101))
plt.ylim(plt.ylim())
max_length = max(all_lengths)
plt.plot([max_length, max_length], plt.ylim())
plt.title(f'Maximum tokens per example: {max_length}');
plt.show()

# Create tokenizer functions
MAX_TOKENS = 128

def filter_max_tokens(pt, en):
  num_tokens = tf.maximum(tf.shape(pt)[1],tf.shape(en)[1])
  return num_tokens < MAX_TOKENS

def tokenize_pairs(pt, en):
  pt = tokenizers.pt.tokenize(pt)
  # Convert from ragged to dense, padding with zeros.
  pt = pt.to_tensor()

  en = tokenizers.en.tokenize(en)
  # Convert from ragged to dense, padding with zeros.
  en = en.to_tensor()
  return pt, en

# Set up a data pipeline with tf.data
# Dataset.cache - keeps the dataset elements in memory after they're loaded
#                 off disk during the first epoch. This will ensure the dataset does not
#                 become a bottleneck while training your model. If your dataset is too
#                 large to fit into memory, you can also use this method to create a performant
#                 on-disk cache.
# Dataset.prefetch - overlaps data preprocessing and model execution while training.
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
      .filter(filter_max_tokens)
      .prefetch(buffer_size=tf.data.AUTOTUNE))

# Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

#############
# Embedding #
#############
# Token embeddings learn to represent each element as a vector/tensor.
# Embeddings represent tokens in a d-dimensional space where tokens with
# similar meaning will be closer to each other. Converting tokens into
# embedding tensors is done with the built-in tf.keras.layers.Embedding layer,
# which is shown in the encoder/decoder sections

#######################
# Positional Encoding #
#######################
# Positional encodings are added to the embeddings to give the model some
# information about the relative position of the tokens in the sentence.
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # Apply the sine function to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # Apply the cosine function to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

# Set the inner-layer dimensionality and the input/output dimensionality.
n, d = 2048, 512

pos_encoding = positional_encoding(position=n, d_model=d)

# Check the shape.
logging.info('> Shape after positional embedding:{}'.format(pos_encoding.shape))

pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot.
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

# Plot the dimensions.
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()

# A point-wise feed-forward network consists of two linear layers (tf.keras.layers.Dense)
# with a ReLU activation in-between:
def point_wise_feed_forward_network(
  d_model, # Input/output dimensionality.
  dff # Inner-layer dimensionality.
  ):

  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # Shape `(batch_size, seq_len, dff)`.
      tf.keras.layers.Dense(d_model)  # Shape `(batch_size, seq_len, d_model)`.
  ])

#####################
# Build the encoder #
#####################
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               dropout_rate=0.1
               ):
    super(EncoderLayer, self).__init__()


    # Multi-head self-attention.
    self.mha = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=d_model, # Size of each attention head for query Q and key K.
        dropout=dropout_rate,
        )
    # Point-wise feed-forward network.
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    # Layer normalization.
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # Dropout for the point-wise feed-forward network.
    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, training, mask):

    # A boolean mask.
    if mask is not None:
      mask1 = mask[:, :, None]
      mask2 = mask[:, None, :]
      attention_mask = mask1 & mask2
    else:
      attention_mask = None

    # Multi-head self-attention output (`tf.keras.layers.MultiHeadAttention `).
    attn_output = self.mha(
        query=x,  # Query Q tensor.
        value=x,  # Value V tensor.
        key=x,  # Key K tensor.
        attention_mask=attention_mask, # A boolean mask that prevents attention to certain positions.
        training=training, # A boolean indicating whether the layer should behave in training mode.
        )

    # Multi-head self-attention output after layer normalization and a residual/skip connection.
    out1 = self.layernorm1(x + attn_output)  # Shape `(batch_size, input_seq_len, d_model)`

    # Point-wise feed-forward network output.
    ffn_output = self.ffn(out1)  # Shape `(batch_size, input_seq_len, d_model)`
    ffn_output = self.dropout1(ffn_output, training=training)
    # Point-wise feed-forward network output after layer normalization and a residual skip connection.
    out2 = self.layernorm2(out1 + ffn_output)  # Shape `(batch_size, input_seq_len, d_model)`.

    return out2

# Test the encoder layer.
# Input: tensors with random distribution; Output: Shape can be verified
sample_encoder_layer = EncoderLayer(d_model=512, num_attention_heads=8, dff=2048)

sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform((2, 3, 512)), training=False, mask=None)

# Print the shape.
# Shape `(batch_size, input_seq_len, d_model)`.
logging.info('> Shape of the Encoder output:{}'.format(sample_encoder_layer_output.shape))

# Create the encoder
# The Transformer encoder consists of:
# Input embeddings (with tf.keras.layers.Embedding)
# Positional encoding (with positional_encoding())
# N encoder layers (with EncoderLayer())
class Encoder(tf.keras.layers.Layer):
  def __init__(self,
               *,
               num_layers,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               input_vocab_size, # Input (Portuguese) vocabulary size.
               dropout_rate=0.1
               ):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    # Embeddings.
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
    # Positional encoding.
    self.pos_encoding = positional_encoding(MAX_TOKENS, self.d_model)

    # Encoder layers.
    self.enc_layers = [
        EncoderLayer(
          d_model=d_model,
          num_attention_heads=num_attention_heads,
          dff=dff,
          dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    # Dropout.
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  # Masking.
  def compute_mask(self, x, previous_mask=None):
    return self.embedding.compute_mask(x, previous_mask)

  def call(self, x, training):

    seq_len = tf.shape(x)[1]

    # Sum up embeddings and positional encoding.
    mask = self.compute_mask(x)
    x = self.embedding(x)  # Shape `(batch_size, input_seq_len, d_model)`.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    # Add dropout.
    x = self.dropout(x, training=training)

    # N encoder layers.
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # Shape `(batch_size, input_seq_len, d_model)`.

# Instantiate the encoder.
sample_encoder = Encoder(
    num_layers=2,
    d_model=512,
    num_attention_heads=8,
    dff=2048,
    input_vocab_size=8500
    )

# Set the test input.
temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

sample_encoder_output = sample_encoder(temp_input,
                                       training=False)

# Print the shape.
# Shape `(batch_size, input_seq_len, d_model)`.
logging.info('> Encoder output shape:{}'.format(sample_encoder_output.shape))

#####################
# Build the Decoder #
#####################
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               dropout_rate=0.1
               ):
    super(DecoderLayer, self).__init__()

    # Masked multi-head self-attention.
    self.mha_masked = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=d_model, # Size of each attention head for query Q and key K.
        dropout=dropout_rate
    )
    # Multi-head cross-attention.
    self.mha_cross = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=d_model, # Size of each attention head for query Q and key K.
        dropout=dropout_rate
    )

    # Point-wise feed-forward network.
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    # Layer normalization.
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # Dropout for the point-wise feed-forward network.
    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, mask, enc_output, enc_mask, training):
    # The encoder output shape is `(batch_size, input_seq_len, d_model)`.

    # A boolean mask.
    self_attention_mask = None
    if mask is not None:
      mask1 = mask[:, :, None]
      mask2 = mask[:, None, :]
      self_attention_mask = mask1 & mask2

    # Masked multi-head self-attention output (`tf.keras.layers.MultiHeadAttention`).
    # First multihead attention. In this query, key and value is target sentence embedding.
    attn_masked, attn_weights_masked = self.mha_masked(
        query=x,
        value=x,
        key=x,
        attention_mask=self_attention_mask,  # A boolean mask that prevents attention to certain positions.
        #use_causal_mask=True,  # A boolean to indicate whether to apply a causal mask to prevent tokens from attending to future tokens.
        return_attention_scores=True,  # Shape `(batch_size, target_seq_len, d_model)`.
        training=training  # A boolean indicating whether the layer should behave in training mode.
        )

    # Masked multi-head self-attention output after layer normalization and a residual/skip connection.
    # Input (target embedding) + first attention layer output.
    out1 = self.layernorm1(attn_masked + x)

    # A boolean mask.
    attention_mask = None
    if mask is not None and enc_mask is not None:
      mask1 = mask[:, :, None]
      mask2 = enc_mask[:, None, :]
      attention_mask = mask1 & mask2

    # Multi-head cross-attention output (`tf.keras.layers.MultiHeadAttention `).
    # Second multihead attention. In this query is the output of the first layer normalization layer.
    # key and value are the context vector generated by encoder.
    attn_cross, attn_weights_cross = self.mha_cross(
        query=out1,
        value=enc_output,
        key=enc_output,
        attention_mask=attention_mask,  # A boolean mask that prevents attention to certain positions.
        return_attention_scores=True,  # Shape `(batch_size, target_seq_len, d_model)`.
        training=training  # A boolean indicating whether the layer should behave in training mode.
    )

    # Multi-head cross-attention output after layer normalization and a residual/skip connection.
    # Output from first layernormalization + output of the second multihead attention.
    out2 = self.layernorm2(attn_cross + out1)  # (batch_size, target_seq_len, d_model)

    # Point-wise feed-forward network output.
    ffn_output = self.ffn(out2)  # Shape `(batch_size, target_seq_len, d_model)`.
    ffn_output = self.dropout1(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # Shape `(batch_size, target_seq_len, d_model)`.

    return out3, attn_weights_masked, attn_weights_cross

# Test the decoder layer.
# Input - tensor with uniform distribution; Output - Verify the shape of the output
sample_decoder_layer = DecoderLayer(d_model=512, num_attention_heads=8, dff=2048)

sample_decoder_layer_output, att1, att2 = sample_decoder_layer(
    x=tf.random.uniform((2, 5, 512)),
    mask=None,
    enc_output=sample_encoder_layer_output,
    enc_mask=None,
    training=False)

# Print the shape.
# `(batch_size, target_seq_len, d_model)`
logging.info('> Output shape of the decoder:{}'.format(sample_decoder_layer_output.shape))

# The Transformer decoder consists of:
# Output embeddings (with tf.keras.layers.Embedding)
# Positional encoding (with positional_encoding())
# N decoder layers (with DecoderLayer)
class Decoder(tf.keras.layers.Layer):
  def __init__(self,
               *,
               num_layers,
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               target_vocab_size,
               dropout_rate=0.1
               ):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(
      target_vocab_size,
      d_model,
      mask_zero=True
      )
    self.pos_encoding = positional_encoding(MAX_TOKENS, d_model)

    self.dec_layers = [
        DecoderLayer(
          d_model=d_model,
          num_attention_heads=num_attention_heads,
          dff=dff,
          dropout_rate=dropout_rate)
        for _ in range(num_layers)
        ]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, enc_output, enc_mask, training):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    # Sum up embeddings and positional encoding.
    mask = self.embedding.compute_mask(x)
    x = self.embedding(x)  # Shape: `(batch_size, target_seq_len, d_model)`.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2  = self.dec_layers[i](x, mask, enc_output, enc_mask, training)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # The shape of x is `(batch_size, target_seq_len, d_model)`.
    return x, attention_weights

# Test the decoder
# Instantiate the decoder.
sample_decoder = Decoder(
    num_layers=2,
    d_model=512,
    num_attention_heads=8,
    dff=2048,
    target_vocab_size=8000
    )

# Set the test input.
temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

output, attn = sample_decoder(
    x=temp_input,
    enc_output=sample_encoder_output,
    enc_mask=None,
    training=False)

# Print the shapes.
logging.info('> Decoder output shape:{}'.format(output.shape))
logging.info('> Attention weights shape:{}'.format(attn['decoder_layer2_block2'].shape))

#########################
# Build the Transformer #
#########################
# Put together the Encoder and Decoder.
class Transformer(tf.keras.Model):
  def __init__(self,
               *,
               num_layers, # Number of decoder layers.
               d_model, # Input/output dimensionality.
               num_attention_heads,
               dff, # Inner-layer dimensionality.
               input_vocab_size, # Input (Portuguese) vocabulary size.
               target_vocab_size, # Target (English) vocabulary size.
               dropout_rate=0.1
               ):
    super().__init__()
    # The encoder.
    self.encoder = Encoder(
      num_layers=num_layers,
      d_model=d_model,
      num_attention_heads=num_attention_heads,
      dff=dff,
      input_vocab_size=input_vocab_size,
      dropout_rate=dropout_rate
      )

    # The decoder.
    self.decoder = Decoder(
      num_layers=num_layers,
      d_model=d_model,
      num_attention_heads=num_attention_heads,
      dff=dff,
      target_vocab_size=target_vocab_size,
      dropout_rate=dropout_rate
      )

    # The final linear layer.
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument.
    # Portuguese is used as the input (`inp`) language.
    # English is the target (`tar`) language.
    inp, tar = inputs

    # The encoder output.
    enc_output = self.encoder(inp, training)  # `(batch_size, inp_seq_len, d_model)`
    enc_mask = self.encoder.compute_mask(inp)

    # The decoder output.
    dec_output, attention_weights = self.decoder(
        tar, enc_output, enc_mask, training)  # `(batch_size, tar_seq_len, d_model)`

    # The final linear layer output.
    final_output = self.final_layer(dec_output)  # Shape `(batch_size, tar_seq_len, target_vocab_size)`.

    # Return the final output and the attention weights.
    return final_output, attention_weights

# Set the hyper parameters
num_layers = 4 # Number of decoder layers
d_model = 128  # Dimensions of input output vector
dff = 512      # Hidden layer units
num_attention_heads = 8 # Number of heads in multihead attention
dropout_rate = 0.1 # dropout rate to control overfit

# Initiate the transformer
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_attention_heads=num_attention_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)

# Test the transformer
input = tf.constant([[1,2,3, 4, 0, 0, 0]])
target = tf.constant([[1,2,3, 0]])

x, attention = transformer((input, target))
logging.info('> Transformer output shape:{}'.format(x.shape))
logging.info('> Attention1 weight shape:{}'.format(attention['decoder_layer1_block1'].shape))
logging.info('> Attention2 weight shape:{}'.format(attention['decoder_layer4_block2'].shape))
transformer.summary()

######################
# Training the model #
######################
# We use a learning rate scheduler as per formula in the paper
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Plot the sample learning rate for 40k steps
temp_learning_rate_schedule = CustomSchedule(d_model)
plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')
plt.show()

# Initiate the Adam optimizer
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# Define the loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

# Save the checkpoint while training
#checkpoint_path = './checkpoints/train'
checkpoint_path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\Checkpoints\5'

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# If a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  logging.info('> Latest checkpoint restored!!')

# Define the training step
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

# The `@tf.function` trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))

EPOCHS = 20
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_batches):
    train_step(inp, tar)

    if batch % 50 == 0:
      logging.info(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    logging.info(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  logging.info(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  logging.info(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

#############
# Inference #
#############
# Steps to create a translation class
# Encode the input sentence using the Portuguese tokenizer (tokenizers.pt). This is the encoder input.
# The decoder input is initialized to the [START] token.
# Calculate the padding masks and the look ahead masks.
# The decoder then outputs the predictions by looking at the encoder output and its own output (self-attention).
# Concatenate the predicted token to the decoder input and pass it to the decoder.
# Auto-regressive: Decoder predicts the next token based on the previous tokens it predicted.
class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS):
    # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # As the output language is English, initialize the output with the
    # English `[START]` token.
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output], training=False)

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.
    text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

    tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # Therefore, recalculate them outside the loop.
    _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

    return text, tokens, attention_weights

# Create an instance of this Translator class
translator = Translator(tokenizers, transformer)

def print_translation(sentence, tokens, ground_truth):
  logging.info(f'{"Input:":15s}: {sentence}')
  logging.info(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  logging.info(f'{"Ground truth":15s}: {ground_truth}')

sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))

print_translation(sentence, translated_text, ground_truth)

# Create a function that plots the attention when a token is generated:
def plot_attention_head(in_tokens, translated_tokens, attention):
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  labels = [label.decode('utf-8') for label in in_tokens.numpy()]
  ax.set_xticklabels(
      labels, rotation=90)

  labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
  ax.set_yticklabels(labels)

# Attention plots per each head
head = 0
# Shape: `(batch=1, num_attention_heads, seq_len_q, seq_len_k)`.
attention_heads = tf.squeeze(
  attention_weights['decoder_layer4_block2'], 0)
attention = attention_heads[head]
logging.info('> Attention head shape:{}'.format(attention.shape))

in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.pt.lookup(in_tokens)[0]
logging.info('> Input portuguese tokens:{}'.format(in_tokens))
logging.info('> Translated english tokens:{}'.format(translated_tokens))

plot_attention_head(in_tokens, translated_tokens, attention)

def plot_attention_weights(sentence, translated_tokens, attention_heads):
  in_tokens = tf.convert_to_tensor([sentence])
  in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
  in_tokens = tokenizers.pt.lookup(in_tokens)[0]

  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)

    plot_attention_head(in_tokens, translated_tokens, head)

    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()

plot_attention_weights(sentence,
                       translated_tokens,
                       attention_weights['decoder_layer4_block2'][0])

# Export the model
# tf.function only the output sentence is returned. Thanks to the
# non-strict execution in tf.function any unnecessary values are never computed.
class ExportTranslator(tf.Module):
  def __init__(self, translator):
    self.translator = translator

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def __call__(self, sentence):
    (result,
     tokens,
     attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)

    return result

translator = ExportTranslator(translator)
modelDir = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\transformer\model'
tf.saved_model.save(translator, export_dir=modelDir)






