import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from methods import missingValues, df_to_dataset, get_normalization_layer, get_category_encoding_layer

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'ModelYear', 'Origin']

# Read the csv formatted data in to an dataframe
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

# Check for missing values
misVal, mis_val_table_ren_columns = missingValues(dataset)
print(misVal.head(10))

# Drop the missing values (for simplicity)
dataset = dataset.dropna()
misVal, mis_val_table_ren_columns = missingValues(dataset)
print(misVal.head(10))

# Separate target and features
dataset['target'] = dataset['MPG']
dataset = dataset.drop(columns=['MPG'])

# Create train, test and validation partitions
train, test = train_test_split(dataset, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Convert the dataframe to a tensorflow dataset
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# # Print the features
# [(train_features, label_features)] = train_ds.take(1)
# print('Every feature:', list(train_features.keys()))
#
# photo_count_col = train_features['Acceleration']
# layer = get_normalization_layer('Acceleration', train_ds)
# print(layer(photo_count_col))

# it = (iter(train_ds))
# train_fds = []
# for t, l in it:
#     tlist = []
#     for tensor in list(t.values()):
#         tlist.append(tensor.numpy())
#     train_fds.append([tlist, l.numpy()])
# train_fds = tf.data.Dataset.from_tensor_slices(train_fds)

all_inputs = []
encoded_features = []

# Categorical features encoded as integers.
for header in ['Cylinders', 'ModelYear']:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
    encoding_layer = get_category_encoding_layer(header, train_ds, dtype='int64',
                                                 max_tokens=5)
    encoded_age_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_age_col)

# Categorical features encoded as string.
categorical_cols = ['Origin']
for header in categorical_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                 max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)

# Numeric features.
for header in ['Displacement', 'Horsepower', 'Weight', 'Acceleration']:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

# Create a functional model.
all_features = tf.keras.layers.Concatenate()(encoded_features)
process_layer = tf.keras.Model(all_inputs, all_features)

# Visualize the model graph
# rankdir='LR' is used to make the graph horizontal.
tf.keras.utils.plot_model(process_layer, show_shapes=True, rankdir="LR")

# Test for a single value
auto_mgp_dict = {name: np.array(value)
                 for name, value in dataset.items()}

features_dict = {name: values[:1] for name, values in auto_mgp_dict.items()}
process_layer(features_dict)

preprocessed_inputs = process_layer(all_inputs)
x = tf.keras.layers.Dense(32, activation="relu")(preprocessed_inputs)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x) #(preprocessed_inputs)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
              loss='mean_absolute_error')
model.summary()

# Train the model
history = model.fit(train_ds,
                    epochs=100
                    # validation_data=val_ds
                    )

# Plot the history and accuracies
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print(hist.tail())

plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, 10])
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True)
plt.show()
