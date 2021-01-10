# Import the relevant packages
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from matplotlib import pyplot as plt
import numpy as np
# import pathlib
import shutil
# import tempfile
import os
from methods import pack_row, ds_shape
from tensorboard import program

print(tf.__version__)

# logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
logdir = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\logs'
tflogdir = os.path.join(logdir, 'tensorboard_logs')
shutil.rmtree(tflogdir, ignore_errors=True)

# Download the dataset if not already
path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Git\Tensorflow-Cert\Exercises\01 Data'
folder = 'nlp'
abs_path = os.path.join(path, folder)
abs_path = os.path.join(abs_path, 'higgs')
if not os.path.exists(os.path.join(abs_path, 'HIGGS.csv.gz')):
    higgs_gz = tf.keras.utils.get_file('HIGGS.csv.gz',
                                       cache_subdir=abs_path,
                                       origin='http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz',
                                       )
    higgs_dir = abs_path
else:
    higgs_dir = abs_path

file = os.path.join(higgs_dir, 'HIGGS.csv.gz')
FEATURES = 28
# Read the csv from the tensorflow method. # not working because of duplicate column names
# higgs_ds = tf.data.experimental.make_csv_dataset(
#             file,
#             batch_size=32,
#             column_defaults=[float(),]*(FEATURES+1),
#             num_epochs=1,
#             compression_type="GZIP"
#         )

higgs_ds = tf.data.experimental.CsvDataset(
    file,
    [float(), ] * (FEATURES + 1),
    compression_type="GZIP"
)

# Get the first 10k records from the dataset
higgs_ds_set = higgs_ds.batch(10000).map(pack_row).unbatch()

# Set the parameters
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

# First 1k for validation an
validate_ds = higgs_ds_set.take(N_VALIDATION).cache()
train_ds = higgs_ds_set.skip(N_VALIDATION).take(N_TRAIN).cache()

ds_shape(train_ds)

# Shuffle and create batches of the dataset
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

# Gradually decrease the learning rate
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


# Set up the learning rate scheduler
step = np.linspace(0, 100000)
lr = lr_schedule(step)
plt.figure(figsize=(8, 6))
plt.plot(step / STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')
plt.show()


# Define the call backs
def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(os.path.join(logdir, name)),
    ]


def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.losses.BinaryCrossentropy(
                          from_logits=True, name='binary_crossentropy'),
                      'accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0)
    return history


# Build a small model for baseline
size_histories = {}
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)

tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

# plotter.plot(size_histories)
# plt.ylim([0.5, 0.7])
# plt.show()

# Now we will increase the model complexity progressively
small_model = tf.keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])

size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

# Medium complexity
medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])

size_histories['Medium']  = compile_and_fit(medium_model, "sizes/Medium")

# complex.
large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])

size_histories['large'] = compile_and_fit(large_model, "sizes/large")

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.show()

# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', os.path.join(logdir, 'sizes')])
# url = tb.launch()
# print(url)

regularizer_histories = {}
regularizer_histories['large'] = size_histories['large']


# Now we will add regularize penalties to the weights
l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")


# Now add drop out layers to avoid overfit
dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")


# Add both regularize weights and dropout layers
combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.show()
