import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from methods import custom_standardization
from tensorflow.keras import losses

path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models'
folder = 'Checkpoints/1/'
cp_path = os.path.join(path, folder)
folder = 'Models/1/'
mp_path = os.path.join(path, folder)

# Restore the weights
# When custom loss function is used
# model = tf.keras.models.load_model(mp_path,
#                                    custom_objects={"HuberLoss": HuberLoss})
model = tf.keras.models.load_model(mp_path)

max_features = 10
sequence_length = 4
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Inference on new data
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

# Save the entire model in HDF5 format
# model.load_weights(cp_path)
tf.saved_model.save(model, mp_path)

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
pred = export_model.predict(examples)
print(pred)

examples_seq = []
vectorize_layer.adapt(examples)
for i in range(len(examples)):
    text = tf.expand_dims(examples[i], -1)
    examples_seq.append(vectorize_layer(text))

print(examples_seq)

pred = model.predict(examples_seq[2])
print(pred)
