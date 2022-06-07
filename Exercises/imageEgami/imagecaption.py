############################################
# Derive a caption for a given image using #
# using visual attention                   #
# Dataset:  MS-COCO dataset                #
############################################

import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# your model focuses on during captioning
import matplotlib.pyplot as plt

import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image

########################################
# Part 1 - Process the images and      #
# use Inception v3 to get the features #
########################################

# Read the annotations file
annotations = r'C:\Users\pmspr\Documents\Machine Learning\Data\Image caption\annotations'
images = r'C:\Users\pmspr\Documents\Machine Learning\Data\Image caption\train2014'
annotation_file = os.path.join(annotations,'captions_train2014.json')
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Group all captions together having the same image ID.
image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
  caption = f"<start> {val['caption']} <end>"
  image_path = images + '\COCO_train2014_' + '%012d.jpg' % (val['image_id'])
  image_path_to_caption[image_path].append(caption)

image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

# Select the first 6000 image_paths from the shuffled set.
# Approximately each image id has 5 captions associated with it, so that will
# lead to 30,000 examples.
train_image_paths = image_paths[:6000]

# Create two separate lists (1-to-1 mapping) for image paths and captions
train_captions = []
img_name_vector = []

for image_path in train_image_paths:
  caption_list = image_path_to_caption[image_path]
  train_captions.extend(caption_list)
  img_name_vector.extend([image_path] * len(caption_list))

# Test the image and its captions
# print(image_path_to_caption[img_name_vector[0]])
# im = Image.open(img_name_vector[0])
# im.show()

# In this we use Inception V3 engine to obtain the features of the image.
# These features are used as source for encoder and decoder model to train the captions.

# convert the images into InceptionV3's expected format by:
# Resizing the image to 299px by 299px
# Preprocess the images using the preprocess_input method to normalize
# the image so that it contains pixels in the range of -1 to 1, which
# matches the format of the images used to train InceptionV3.
  def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# create a tf.keras model  The shape of the output of this layer is 8x8x2048
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input

# where the output layer is the last convolutional layer in the InceptionV3 architecture.
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)

# Map pre-processing method, load_image, to each image
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

for img, path in image_dataset:
  # You forward each image through the network
  # (16, 8, 8, 2048) - batch-size, output from conv layer - 8*8*2048
  batch_features = image_features_extract_model(img)

  # (16, 64, 2048) - batch-size, -1 flattens 8*8 to 64, 2048
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  # store the resulting vector in a dictionary (image_name --> feature_vector).
  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    # After all the images are passed through the network, you save the dictionary to disk.
    np.save(path_of_feature, bf.numpy())

##############################################
# Part 2 - Process the Captions and          #
# use encoder-decoer with attentions for NMT #
##############################################

# Preprocess and tokenize the captions

# Create a tensorflow dataset using the train captions
caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

# We will override the default standardization of TextVectorization to preserve
# "<>" characters, so we preserve the tokens for the <start> and <end>.
def standardize(inputs):
  inputs = tf.strings.lower(inputs)
  return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")

# Max word count for a caption. All output sequences will be padded to length 50.
max_length = 50

# Use the top 5000 words for a vocabulary. Compute a vocabulary of the top 5,000 words (to save memory).
vocabulary_size = 5000

# Tokenize all captions by mapping each word to it's index in the vocabulary.
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize=standardize,
    output_sequence_length=max_length)

# Learn the vocabulary from the caption data.
# Use adapt to iterate over all captions, split the captions into words
tokenizer.adapt(caption_dataset)

# Create the tokenized vectors
cap_vector = caption_dataset.map(lambda x: tokenizer(x))

# Create mappings for words to indices and indicies to words.
# Create word-to-index mappings
word_to_index = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())

# Create index-to-word mappings
index_to_word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

# At this point we have,
# 1. A dictory of image paths and their corresponding captions.
# 2. A model to get the incetion v3 features for a given batch of images
# 3. A preprocessing layer for captions.

# Split the data into training and testing

# Create a dict to capture an image and its tokenized vector
img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
  img_to_cap_vector[img].append(cap)

# Create training and validation sets using an 80-20 split randomly.
img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

# Create 80:20 split for Training:Validation
slice_index = int(len(img_keys)*0.8)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

# Create two lists having 1-to-1 mapping for training
# with image path and its caption. Remember one image can have multiple captions.
img_name_train = []
cap_train = []
for imgt in img_name_train_keys:
  capt_len = len(img_to_cap_vector[imgt])
  img_name_train.extend([imgt] * capt_len)
  cap_train.extend(img_to_cap_vector[imgt])

# Create two lists having 1-to-1 mapping for validation
img_name_val = []
cap_val = []
for imgv in img_name_val_keys:
  capv_len = len(img_to_cap_vector[imgv])
  img_name_val.extend([imgv] * capv_len)
  cap_val.extend(img_to_cap_vector[imgv])

print('Training and validatinon lengths:')
print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

# Create a tf.data dataset for training
# We have a tuple with Image path and its caption. Now we replace image path with its image tensor
# obtained from inception v3 engine.

# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int64]),
          num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
    #print(f'Features shape, (batch_size, 64, embedding_dim):{features.shape}')

    # hidden shape == (batch_size, hidden_size)
    #print(f'Hidden state shape,(batch_size, hidden_size):{hidden.shape} ')

    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    #print(f'Hidden state with time step shape,(batch_size, 1, hidden_size):{hidden_with_time_axis.shape} ')

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))
    #print(f'Attention layer shape,(batch_size, 64, units):{attention_hidden_layer.shape}')

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)
    #print(f'Attention score shape, (batch_size, 64, 1): {score.shape}')

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, embed_size)
    context_vector = attention_weights * features
    #print(f'Context vector shape,(batch_size, 64, embed_size):{context_vector.shape}')

    context_vector = tf.reduce_sum(context_vector, axis=1)
    #print(f'Context vector shape After reduce,(batch_size, embed_size):{context_vector.shape}')

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

# Test attention
# hidden = tf.zeros((BATCH_SIZE, units))
# encoder = CNN_Encoder(embedding_dim)
# for (batch, (img_tensor, target)) in enumerate(dataset):
#     features = encoder(img_tensor)
#     break
# attention = BahdanauAttention(units)
# context_vector, attention_weights = attention(features, hidden)

# Above test of attention layer for a single tensor prints below statements
# Features shape, (batch_size, 64, embedding_dim):(64, 64, 256)
# - Our features are 8*8*2048 matrix from Inception v3 engine
# - 8*8 image patch (like a word in a sentence) - each pixel is represented by 2048 vetor.
# - We squash 8*8 in to 64 - (64, 2048).
# - We send this (64, 2048) through an embedding layer of 256 dimension making it (64, 256)
# - For a batch size=64, we get final feature shape as - (64, 64, 256)

# Hidden state shape,(batch_size, hidden_size):(64, 512)
# - Here state is of the decoder. For each time step it generate a hidden state basing on the number
#   of RNN (here GRU) units = 512
# - As we are not yet at decoding, we fabricated a zero vector of (64, 512).

# Hidden state with time step shape,(batch_size, 1, hidden_size):(64, 1, 512)
# - We add another dimension for the time step. (64, 512) -> (64, 1, 512)
#   like for one word in the sentence (a time step in the unrolling) is represented by 512 vector and for
#   all the sentences in that batch

# Attention layer shape,(batch_size, 64, units):(64, 64, 512)
# - According to the paper, we use global and soft attention - "the hidden activation of LSTM is a linear
#   projection of of the stochastic context vector followed by tanh non-linearity"
# - Send features through a dense layer of size 512 units.
# - W1*features = (64,64,256) - (64,64,512)
# - Send hidden state through a dense layer of size 512 units.
# - W2*hidden state = (64,1,512) - (64,1,512)
# - attention hidden layer = tanh(W1*features + W2*hiddenstate) = (64,64,512)

# Attention score shape, (batch_size, 64, 1): (64, 64, 1)
# - Send attention hidden layer through a dense layer of size one unit
# - This is like getting a value (score) for each word (here size of sentence is 64)
#   for every batch of size 64.
#   [ [ [score],[score],[score]...64(annotation vector size) ],[],[]...64(batch_size) ]

# Context vector shape,(batch_size, embed_size):(64, 64, 256)
# - This product of attention weights and features
# - (64,64,1)*(64,64,256) -> (64,64,256)

# Context vector shape After reduce,(batch_size, embed_size):(64, 256)
# - We add values for axis=1, that is for second dimension
# - (64,64,256) -> (64,256)

###########
# Decoder #
###########
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)
    #print(f'Shape of target after embedding:{x.shape}')

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    #print(f'Shape of target after concatinating embed vector and hidden state:{x.shape}')

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)
    #print(f'Shape of the output:{output.shape}')
    #print(f'Shape of the state:{state.shape}')

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)
    #print(f'Output share from dense layer:{x.shape}')

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))
    #print(f'Output share from dense layer after reshape:{x.shape}')

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)
    #print(f'Output share from dense layer with logits:{x.shape}')

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

# Instantiate encoder and decoder objects
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())

# Test the decoder
# for (batch, (img_tensor, target)) in enumerate(dataset):
#     features = encoder(img_tensor)
#     break
# hidden = decoder.reset_state(batch_size=target.shape[0])
# dec_input = tf.expand_dims([word_to_index('<start>')] * target.shape[0], 1)
# predictions, hidden, _ = decoder(dec_input, features, hidden)

# Shape of target after embedding:(64, 1, 256)
# - Embedding dimension = 256, one target sentence is tokenized in to a vector
# - Tokenized vector is  then converted to embedded vector of dimension = embed_size = 256
# - One batch of 64 sentences. One sentence represented as a vector of size 256.
# [ [ [1,2,3..256] ],
#   [ [1,2,3..256] ],
#   [ [1,2,3..256] ]....64]

# Shape of target after concatinating embed vector and hidden state:(64, 1, 512)
# - In the paper it is mentioned - "In this work, we use a deep output layer (Pascanu et al.,
# 2014) to compute the output word probability given the
# LSTM state, the context vector and the previous word"
# We concatenate and current (time step) target and context vector (which is the result of attention).
# Attention is consider previous hidden state.
# - Target size = (64, 1, 256) (after embedding)
# - Context vector = (64, 256)=> tf.expand_dims(context_vector, 1) = (64,1,256)
# - tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1) = (64, 1, 256+256) = (64,1,512)

# Shape of the output:(64, 1, 512)
# - GRU has 512 units.
# Input - (64,1,512) ; Output - (64,1,512)

# Shape of the state:(64, 512)
# - State is the gist of entire time step.

# Output share from dense layer:(64, 1, 512)
# - Dense layer fc1 has 512 units.
# - Input (64,1,512); Output - (64,1,512)

# Output share from dense layer after reshape:(64, 512)
# - We are trying to predict the conditional probability of the given target word.
# - Decoder predicts a word, for one time step, taking the context vector and input features.
# - This dense layer provide a vector of 512 for each image in the batch

# Output share from dense layer with logits:(64, 5000)
# - We have a corpus of maximum 5000 words. We use a beam search or softmax to
#   deduce the word with maximum probablity.
# - We use another dense to ouput a vector of 5000 for each image in the batch.

# Define the optimizer.
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


# Define the loss function.
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))

  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

# Logic to set checkpoints during training.
checkpoint_path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\Checkpoints\4'
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)

##############
# Training   #
##############
# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []

@tf.function
def train_step(img_tensor, target):
  loss = 0

  # You extract the features stored in the respective .npy files - Image tensor

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  #the  decoder input(which is the start token) is passed to the decoder.
  dec_input = tf.expand_dims([word_to_index('<start>')] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder.
          # The decoder returns the predictions and the decoder hidden state.
          # The decoder hidden state is then passed back into the model
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          #print(f'Target shape:{target.shape}')
          #print(f'Predictions shape:{predictions.shape}')

          # predictions are used to calculate the loss.
          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          # Use teacher forcing to decide the next input to the decoder.
          # Teacher forcing is the technique where the target word is passed as the next input to the decoder.
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)
  # The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss

# Test the Train step
# Total loss for test Train step:1.8630383014678955
# total_loss = 0
# for (batch, (img_tensor, target)) in enumerate(dataset):
#     batch_loss, t_loss = train_step(img_tensor, target)
#     total_loss += t_loss
#     break
# print(f'Total loss for test Train step:{total_loss}')

# Target shape:(64, 50)
# - Batch-size = 64; size of the target vector - 50
# - We send integer by integer from the target vector and aggregate the loss.

# Predictions shape:(64, 5000)
# - Batch_size = 64; Each patch is given probabilities (logits) for all the classes (5000)
# - We use SparseCategoricalCrossentropy as
#   The shape of y_true is [batch_size]
#   The shape of y_pred is [batch_size, num_classes]
# - Y-pred is out of a dense layer (no softmax), so it is expected to be a logits tensor.

EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            average_batch_loss = batch_loss.numpy()/int(target.shape[1])
            print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
      ckpt_manager.save()

    print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

class Translator(tf.Module):

  def __init__(self, encoder, decoder):
    self.encoder = encoder
    self.decoder = decoder

translator = Translator(
    encoder= encoder,
    decoder= decoder
)

def evaluate(self, image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = self.decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = self.encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = self.decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

Translator.evaluate = evaluate

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(int(np.ceil(len_result/2)), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

# Test the translator
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tf.compat.as_text(index_to_word(i).numpy())
                         for i in cap_val[rid] if i not in [0]])
#tf.config.run_functions_eagerly(False)
result, attention_plot = translator.evaluate(image = image)
print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)

# Tried to save the model. did not work
#mpath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\Models\6'
#tf.saved_model.save(translator, mpath)#,
                    #signatures={'serving_default': translator.evaluate})

# reloaded = tf.saved_model.load(mpath)
# rid = np.random.randint(0, len(img_name_val))
# image = img_name_val[rid]
# real_caption = ' '.join([tf.compat.as_text(index_to_word(i).numpy())
#                          for i in cap_val[rid] if i not in [0]])
# result, attention_plot = reloaded.evaluate(image = image)
# print('Real Caption:', real_caption)
# print('Prediction Caption:', ' '.join(result))

