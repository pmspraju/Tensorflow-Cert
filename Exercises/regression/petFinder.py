import numpy as np
import pandas as pd
import tensorflow as tf
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from methods import df_to_dataset, get_normalization_layer, get_category_encoding_layer

# Download the dataset if not already
path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Git\Tensorflow-Cert\Exercises\01 Data'
folder = 'nlp'
abs_path = os.path.join(path, folder)
if not os.path.exists(os.path.join(abs_path, 'petfinder-mini')):
    pf_text_r = tf.keras.utils.get_file('petfinder-mini.zip',
                                        untar=True,
                                        cache_subdir=abs_path,
                                        origin='http://storage.googleapis.com/download.tensorflow.org/data'
                                               '/petfinder-mini.zip',
                                        extract=True)
    pf_dir = os.path.join(abs_path, 'petfinder-mini')
else:
    pf_dir = os.path.join(abs_path, 'petfinder-mini')

file = os.path.join(pf_dir, 'petfinder-mini.csv')
dataframe = pd.read_csv(file)

# In the original dataset "4" indicates the pet was not adopted.
dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4, 0, 1)

# Drop un-used columns.
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)

[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['Age'])
print('A batch of targets:', label_batch)


###################################################
# Different ways to traverse through the dataset  #
# can be used to understand the data              #
###################################################
# Method to create generator for the dataset
def slices(features):
    for i in itertools.count():
        # For each feature take index `i`
        example = {name: values[i] for name, values in features.items()}
        yield example


# Create a dictionary from the dataset
# Test for a single value
pet_finder_dict = {name: np.array(value)
                   for name, value in dataframe.items()}

# Using above generator, view the data
for example in slices(pet_finder_dict):
    for name, value in example.items():
        print(f"{name:19s}: {value}")
    break

# Create a tensorflow dataset from slices of dictionary
features_ds = tf.data.Dataset.from_tensor_slices(pet_finder_dict)

# Print from the slices
for example in features_ds:
    for name, value in example.items():
        print(f"{name:19s}: {value}")
    break

# Read the csv from the tensorflow method
pet_finder_ds = tf.data.experimental.make_csv_dataset(
    file,
    batch_size=32,
    label_name='AdoptionSpeed',
    num_epochs=1)

for batch, label in pet_finder_ds.take(1):
    for key, value in batch.items():
        print(f"{key:20s}: {value[:5]}")
    print()
    print(f"{'label':20s}: {label[:5]}")
###################################################
photo_count_col = train_features['PhotoAmt']
layer = get_normalization_layer('PhotoAmt', train_ds)
layer(photo_count_col)

type_col = train_features['Type']
layer = get_category_encoding_layer('Type', train_ds, 'string')
layer(type_col)

type_col = train_features['Age']
category_encoding_layer = get_category_encoding_layer('Age', train_ds,
                                                      'int64', 5)
category_encoding_layer(type_col)

batch_size = 256
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

all_inputs = []
encoded_features = []

# Numeric features.
for header in ['PhotoAmt', 'Fee']:
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)

# Categorical features encoded as integers.
age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')
encoding_layer = get_category_encoding_layer('Age', train_ds, dtype='int64',
                                             max_tokens=5)
encoded_age_col = encoding_layer(age_col)
all_inputs.append(age_col)
encoded_features.append(encoded_age_col)

# Categorical features encoded as string.
categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                    'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
for header in categorical_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                 max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)

# Create a functional model.
all_features = tf.keras.layers.Concatenate()(encoded_features)
process_layer = tf.keras.Model(all_inputs, all_features)
preprocessed_inputs = process_layer(all_inputs)

preprocessed_inputs = process_layer(all_inputs)
x = tf.keras.layers.Dense(32, activation="relu")(preprocessed_inputs)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(all_inputs, output)
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

# rankdir='LR' is used to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# Train the model
history = model.fit(train_ds,
                    epochs=100
                    # validation_data=val_ds
                    )

# Plot the history and accuracies
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
# print(hist.tail())

plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, 10])
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True)
plt.show()
