import re
import string
import tensorflow as tf
import pandas as pd
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


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
        print(sum(1 for _ in it))
    except Exception as ex:
        print("-----------------------------------------------------------------------")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
