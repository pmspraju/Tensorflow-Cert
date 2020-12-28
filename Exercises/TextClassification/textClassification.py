# Import relevant packages.
import os
import shutil
import random
import tensorflow as tf
import numpy as np

# from tensorflow.keras import layers
from tensorflow.keras import losses
# from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt
from methods import custom_standardization, HuberLoss

# Download the dataset if not already
path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Git\Tensorflow-Cert\Exercises\01 Data'
folder = 'nlp'
abs_path = os.path.join(path, folder)
if not os.path.exists(os.path.join(abs_path, 'aclImdb')):
    imdb_text_r = tf.keras.utils.get_file('aclImdb_v1.tar.gz',
                                          untar=True,
                                          cache_subdir=abs_path,
                                          origin='https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                                          extract=True)
    os.remove(imdb_text_r)
    imdb_text = abs_path
else:
    imdb_text = abs_path

imdb_dir = os.path.join(imdb_text, 'aclImdb')
# print(os.listdir(imdb_dir))
train_dir = os.path.join(imdb_dir, 'train')
test_dir = os.path.join(imdb_dir, 'test')

# See a random sample review
file_list = os.listdir(os.path.join(train_dir, 'pos'))
file_name = random.sample(file_list, 1)
file_name = 'pos/' + str(file_name[0])
with open(os.path.join(train_dir, file_name)) as f:
    print(f.read())

# Creating dataset from the directory.
# Remove unwanted folders.
remove_dir = os.path.join(train_dir, 'unsup')
if os.path.exists(remove_dir):
    shutil.rmtree(remove_dir)

# Create a text dataset.
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

# See sample texts and their labels
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

# our label_mode of the dataset is inferred. See the classs
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

# Create a validation subset
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)
# print(len(raw_val_ds))
# print(len(raw_train_ds))

# Create the test dataset
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size)

max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# ttl = list(train_text.as_numpy_iterator())
# print(ttl[0][0])

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), float(label)


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))

# Reverse map the tokens and sequences
# print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
# print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# Apply vectorization function to train, validation and test datasets
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Perform data fetching optimization using Autotune.
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Model
embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

# Compile the model with a custom loss function
# model.compile(loss=HuberLoss(2.),
#               optimizer='adam',
#               metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

#Compile using a standard loss function
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Train the model for 10 epochs
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Plot the model's accuracy
history_dict = history.history
# history_dict.keys()
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# Plot the loss
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plot the Accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# Download the dataset if not already
model_copy = model
path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models'
folder = 'Checkpoints/1/'
cp_path = os.path.join(path, folder)
folder = 'Models/1/'
mp_path = os.path.join(path, folder)

# Save the weights
model.save_weights(cp_path)

# Restore the weights
model.load_weights(cp_path)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = model.evaluate(test_ds)
print(accuracy)

# Save the entire model in HDF5 format
tf.saved_model.save(model_copy, mp_path)

# Restore the weights
# When custom loss function is used
# model = tf.keras.models.load_model(mp_path,
#                                    custom_objects={"HuberLoss": HuberLoss})
model = tf.keras.models.load_model(mp_path)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = model.evaluate(test_ds)
print(accuracy)


