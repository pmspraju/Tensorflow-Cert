import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt
from methods import ds_shape, TimeSeriesModel

import numpy as np
import pandas as pd
import sys
import os
import time
import re
import string

print(tf.__version__)
print(sys.version)

# Read the dataset
# Download the dataset if not already
path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Git\Tensorflow-Cert\Exercises\01 Data'
folder = 'ts'
abs_path = os.path.join(path, folder)
abs_path = os.path.join(abs_path, 'temp')
if not os.path.exists(os.path.join(abs_path, 'daily-min-temperatures.csv')):
    tem_ser_r = tf.keras.utils.get_file('daily-min-temperatures.csv',
                                        cache_subdir=abs_path,
                                        origin='https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv',
                                        extract=True)
    tempr_dir = abs_path
else:
    tempr_dir = abs_path

file = os.path.join(tempr_dir, 'daily-min-temperatures.csv')

# Create a pandas dataframe
data = pd.read_csv(file)
print(data.columns)
print(f"Length of the series:{len(data['Temp'])}")

# Create series for plot
series = np.array(data['Temp'])
tfseries = tf.convert_to_tensor(series, dtype=tf.float32)
time = tf.linspace(1, len(series), len(series), name="timesteps")
print(time)


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


plt.figure(figsize=(10, 6))
plot_series(time, tfseries)
plt.show()


# Method to window the dataset
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    # for wd in ds:
    #     print('Window')
    #     for val in wd:
    #         print(val.numpy())
    #     break
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    # for wd in ds:
    #     print(wd.numpy())
    #     break
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    # for x, y in ds:
    #     print('Window example')
    #     print(x.numpy(), y.numpy())
    #     break
    return ds.batch(batch_size).prefetch(1)


# Split the data in to training and validation
split_time = 2500
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Set the parameters
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)
window_size = 64
batch_size = 256
shuffle_buffer_size = 1000

# Create the windowed dataset
train_set = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)
print(train_set)
print(x_train.shape)


# See the data
ds_shape(train_set)
# print('Sample window after batching')
# for feature, label in train_set.take(1):
#     for row in feature:
#         print(row.numpy())
#         break
# print()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)

# Pre defined model using Sequential API
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv1D(filters=60, kernel_size=5,
#                       strides=1, padding="causal",
#                       activation="relu",
#                       input_shape=[None, 1]),
#   tf.keras.layers.LSTM(60, return_sequences=True),
#   tf.keras.layers.LSTM(60, return_sequences=True),
#   tf.keras.layers.Dense(30, activation="relu"),
#   tf.keras.layers.Dense(10, activation="relu"),
#   tf.keras.layers.Dense(1),
#   tf.keras.layers.Lambda(lambda x: x * 400)
# ])
#
# model.compile(loss=tf.keras.losses.Huber(),
#               optimizer=optimizer,
#               metrics=["mae"])
#
# tf.keras.utils.plot_model(model, "model.png", show_shapes=True, rankdir="TB")
# model.summary()
# history = model.fit(train_set,epochs=150)
# history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 60])
# plt.show()


# Model using Functional API
# inputs = tf.keras.Input(shape=(None, 1))
# x = tf.keras.layers.Conv1D(filters=32, kernel_size=5,
#                       strides=1, padding="causal",
#                       activation="relu",
#                       input_shape=[None, 1])(inputs)
# x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
# x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
# x = tf.keras.layers.Dense(30, activation="relu")(x)
# x = tf.keras.layers.Dense(10, activation="relu")(x)
# x = tf.keras.layers.Dense(1)(x)
# outputs = tf.keras.layers.Lambda(lambda y: y * 400)(x)
# fmodel = tf.keras.Model(inputs=inputs, outputs=outputs, name="TimeSeriesModel")
#
# fmodel.compile(loss=tf.keras.losses.Huber(),
#               optimizer=optimizer,
#               metrics=["mae"])
#
# history = fmodel.fit(train_set, epochs=100, callbacks=[lr_schedule])
# tf.keras.utils.plot_model(fmodel, "functional_model.png", show_shapes=True, rankdir="TB")
# fmodel.summary()


# Custom model of above pre define model
cmodel = TimeSeriesModel(
    filters=60,
    kernel_size=5,
    strides=1,
    padding="causal",
    activation="relu",
    return_sequences=True
)

cmodel.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

#history = cmodel.fit(train_set, epochs=150, callbacks=[lr_schedule])
history = cmodel.fit(train_set,epochs=150)

tf.keras.utils.plot_model(cmodel.build_graph(), "custom_model.png", show_shapes=True, rankdir="TB")
cmodel.summary()
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 60])
# plt.show()


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


rnn_forecast = model_forecast(cmodel, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
plt.show()

time_epoch = tf.linspace(1, 150, 150, name="epochsteps")
plt.figure(figsize=(10,6))
plot_series(time_epoch, history.history["loss"])
plot_series(time_epoch, history.history["mae"])
plt.show()