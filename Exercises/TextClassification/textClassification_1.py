############################################
# Text classification with pretrained BERT #
############################################

# import relevant packages
import os
import sys
import random
import shutil
import tensorflow as tf
import tensorflow_hub as hub # pip install tensorflow-hub
import tensorflow_text as text
from official.nlp import optimization  # pip install tf-models-official # to create AdamW optimizer
import matplotlib.pyplot as plt

import constants as cn

tf.get_logger().setLevel('ERROR')

# Read the imdb dataset
path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data\nlp\aclImdb'
mpath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\Models\2'

train_dir = os.path.join(path,'train')
test_dir = os.path.join(path, 'test')
# remove unused folders to make it easier to load the data
#remove_dir = os.path.join(train_dir, 'unsup')
#shutil.rmtree(remove_dir)

# Create train and validation datasets
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

# Create the raw dataset from the directory
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

class_names = raw_train_ds.class_names

# Train and validation dataset
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Example reviews
for review_batch,label_batch in train_ds.take(1):
    for i in range(3):
        print('-------------')
        print('{} review: {}'.format('negative'if(label_batch.numpy()[i] == 0) else 'positive', review_batch.numpy()[i]))

# Choose a BERT model
bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
tfhub_handle_encoder = cn.map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = cn.map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

# Pre processing model for BERT
# TF Hub provides the preprocess model
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

# Feed this pre-process model to BERT model
text_test = ['I loved this movie!']
text_preprocessed = bert_preprocess_model(text_test)
print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
#print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')

# User BERT model. Feed pre-processing directly to bert as
# preprocessing model is a TF model
bert_model = hub.KerasLayer(tfhub_handle_encoder)
bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
#print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
#print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
#print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

'''
The BERT models return a map with 3 important keys: pooled_output, sequence_output, encoder_outputs:
pooled_output represents each input sequence as a whole. The shape is [batch_size, H]. 
You can think of this as an embedding for the entire movie review.

sequence_output represents each input token in the context. 
The shape is [batch_size, seq_length, H]. You can think of this as a contextual embedding 
for every token in the movie review.

encoder_outputs are the intermediate activations of the L Transformer blocks. 
outputs["encoder_outputs"][i] is a Tensor of shape [batch_size, seq_length, 1024] with the 
outputs of the i-th Transformer block, for 0 <= i < L. The last value of the list is equal 
to sequence_output.
'''
# Build custom fine-tune model using BERT
def buildModel():
    input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocess_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocess')
    encoder_input = preprocess_layer(input)
    bert_encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='bert')
    outputs = bert_encoder(encoder_input)
    x = outputs['pooled_output']
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(1, activation=None, name='classifier')(x)
    return tf.keras.Model(input, x)

# Test the model with sample input
classifier_model = buildModel()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

# Plot the model
tf.keras.utils.plot_model(classifier_model)

#loss, metrics
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

#Optimizer
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps) # 10% of training steps

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

# classifier_model.compile(optimizer=optimizer,
#                          loss=loss,
#                          metrics=metrics)
#
# print(f'Training model with {tfhub_handle_encoder}')
# history = classifier_model.fit(x=train_ds,
#                                validation_data=val_ds,
#                                epochs=epochs)
#
saveModel = os.path.join(mpath,'rnn_3')
# classifier_model.save(saveModel, include_optimizer=False)
# loss, accuracy = classifier_model.evaluate(test_ds)
#
# print(f'Loss: {loss}')
# print(f'Accuracy: {accuracy}')

reloaded_model = tf.saved_model.load(saveModel)
examples = [
    'this is such an amazing movie!',  # this is the same sentence tried earlier
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
print(reloaded_results)

