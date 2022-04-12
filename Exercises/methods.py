import re
import string
import tensorflow as tf
import pandas as pd
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from random import randint
import os

# A method to clean the text from punctuation
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


# A method to derive missing values percentages
def missingValues(data):
    try:
        # Total missing values
        mis_val = data.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * mis_val / len(data)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})
        mis_val_table_ren_columns.head(4)
        # Sort the table by percentage of missing descending
        misVal = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

        return misVal, mis_val_table_ren_columns

    except Exception as ex:
        print("-----------------------------------------------------------------------")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    try:
        dataframe = dataframe.copy()
        labels = dataframe.pop('target')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
            ds = ds.batch(batch_size)
            ds = ds.prefetch(batch_size)
        return ds
    except Exception as ex:
        print("-----------------------------------------------------------------------")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


# A method to normalize numeric features
def get_normalization_layer(name, dataset):
    try:
        # Create a Normalization layer for our feature.
        normalizer = preprocessing.Normalization()

        # Prepare a Dataset that only yields our feature.
        feature_ds = dataset.map(lambda x, y: x[name])
        # print(tf.data.DatasetSpec.from_value(feature_ds))
        # ttl = list(feature_ds.as_numpy_iterator())
        # print(len(ttl))
        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)

        return normalizer
    except Exception as ex:
        print("-----------------------------------------------------------------------")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


# A method to encode categorical features both numeric and string
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    try:
        # Create a StringLookup layer which will turn strings into integer indices
        if dtype == 'string':
            index = preprocessing.StringLookup(max_tokens=max_tokens)
        else:
            index = preprocessing.IntegerLookup(max_values=max_tokens)

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        # ttl = list(feature_ds.as_numpy_iterator())
        # print(ttl[0])

        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)

        # Create a Discretization for our integer indices.
        encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

        # Prepare a Dataset that only yields our feature.
        feature_ds = feature_ds.map(index)
        # ttl = list(feature_ds.as_numpy_iterator())
        # print(ttl[0])

        # Learn the space of possible indices.
        encoder.adapt(feature_ds)

        # Apply one-hot encoding to our indices. The lambda function captures the
        # layer so we can use them, or include them in the functional model later.
        return lambda feature: encoder(index(feature))

    except Exception as ex:
        print("-----------------------------------------------------------------------")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


# A custom class to derive huber loss
class HuberLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


def pack_row(*row):
    try:
        label = row[0]
        features = tf.stack(row[1:], 1)
        return features, label
    except Exception as ex:
        print("-----------------------------------------------------------------------")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


def ds_shape(dataset):
    try:
        # Get the number of Features.
        for row in dataset.take(1):
            print('Number of features', len(row))
        # Get the number of examples. Too large dataset
        it = (iter(dataset))
        print('Number of examples', sum(1 for _ in it))
    except Exception as ex:
        print("-----------------------------------------------------------------------")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)


class RnnModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "" or "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
        print(self.prediction_mask.numpy())

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


class CustomTraining(RnnModel):
    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss': loss}


class TimeSeriesModel(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding, activation, return_sequences):
        super().__init__(self)
        self.conv1D = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      activation=activation,
                      input_shape=[None, 1])
        self.lstm = tf.keras.layers.LSTM(60, return_sequences=return_sequences)
        self.lstm1 = tf.keras.layers.LSTM(60, return_sequences=return_sequences)
        self.dense1 = tf.keras.layers.Dense(30, activation=activation)
        self.dense2 = tf.keras.layers.Dense(10, activation=activation)
        self.dense3 = tf.keras.layers.Dense(1)
        self.lmbda = tf.keras.layers.Lambda(lambda y: y * 400)

    def call(self, inputs, training=False):
        x = self.conv1D(inputs, training=training)
        x = self.lstm(x, training=training)
        x = self.lstm1(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)
        x = self.lmbda(x, training=training)

        return x

    def build_graph(self):
        inp = tf.keras.Input(shape=(None, 1))
        return tf.keras.Model(inputs=[inp], outputs=self.call(inp))

def visIntermediate(model,img_path):
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    successive_outputs = [layer.output for layer in model.layers[1:]]

    # Get the layers
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
    # Change image to array and shape
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Normalize the image
    x = x/255.0

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # -----------------------------------------------------------------------
    # Now let's display our representations
    # -----------------------------------------------------------------------

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # -------------------------------------------
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            # -------------------------------------------
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)

            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # -------------------------------------------------
            # Postprocess the feature to be visually palatable
            # -------------------------------------------------
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                #x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size: (i + 1) * size] = x  # Tile each filter into a horizontal grid

            # -----------------
            # Display the grid
            # -----------------
            scale = 20. / n_features

            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()

def dispImg(ipath):
    img = mpimg.imread(ipath)
    plt.imshow(img)
    plt.show()

# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True

def layerOutputs(model, sample_image, ipath):
    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]

    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(sample_image[tf.newaxis, ...])

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    for layerName, feature in zip(layer_names, successive_feature_maps):

        if('conv2d' not in layerName):
            continue

        n_features = feature.shape[-1] # number of feature maps
        size       = feature.shape[1] # size of each feature map
        grid_size  = np.zeros((size, size * n_features))

        for i in range(n_features):
            x = feature[0,:,:,i]
            x -= x.mean()
            #x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            grid_size[:, i * size: (i + 1) * size] = x  # Tile each filter into a horizontal grid

        scale = 20. / n_features
        f = plt.figure(figsize=(scale * n_features, scale))
        plt.title(layerName)
        plt.grid(False)
        plt.imshow(grid_size, aspect='auto', cmap='viridis')
        #plt.show()
        iname = layerName + '.png'
        f.savefig(os.path.join(ipath, iname), bbox_inches='tight')
        plt.close(f)
