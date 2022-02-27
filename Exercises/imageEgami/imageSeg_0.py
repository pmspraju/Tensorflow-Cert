#################################################################
# Segmentation with U-net                                       #
# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ #
#################################################################
# import the packages
import os
import sys
import random
import constants as cn
from methods import myCallback

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Download he datasets from tfds
dataset, info = tfds.load(cn.oxfort_dataset, with_info=True)

# Load the image from the dataset
def load_image(datapoint):

    # resize to proper size
    input_image = tf.image.resize(datapoint['image'], (256,256))
    input_mask  = tf.image.resize(datapoint['segmentation_mask'], (128,128))

    #normalize
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000

# Split in to test and train sets
train_dataset = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# Build a class for augmentation to include in the pipeline
class Augment(tf.keras.layers.Layer):

    def __init__(self, seed=42):
        super(Augment, self).__init__()
        self.inputs_augment = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.labels_augment = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.inputs_augment(inputs)
        labels = self.labels_augment(labels)
        return inputs, labels

# Build the pipeline
train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
test_batches = test_dataset.batch(BATCH_SIZE)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i] + ' with shape ' +str(display_list[i].shape.as_list()))
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

# Build the model
# U-net has convolution blocks, up-convolution blocks joined by concatenate layers
# uses kernel initializer with weights having Gaussian distribution with sd = sqrt(2/N)
# Refer to https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ to architecture.

# use subclassing to build convolution block as a custom layer

# This block will have two convolution, dropout layers
class conv2DBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, kernel_size, drop_rate, **kwargs):
        super(conv2DBlock, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.drop_rate   = drop_rate
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=np.sqrt(2 / (kernel_size ** 2 * filter_size)))
        self.convlayer1 = tf.keras.layers.Conv2D(filters = filter_size,
                                             kernel_size = (3,3),
                                             strides=1,
                                             kernel_initializer = self.initializer,
                                             padding = 'valid'
                                             )
        self.droplayer1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.actlayer1  = tf.keras.layers.Activation('relu')

        self.convlayer2 = tf.keras.layers.Conv2D(filters=filter_size,
                                             kernel_size=(3, 3),
                                             strides=1,
                                             kernel_initializer=self.initializer,
                                             padding='valid'
                                             )
        self.droplayer2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.actlayer2 = tf.keras.layers.Activation('relu')

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.convlayer1(x)
        if (training):
            x = self.droplayer1(x)
        x = self.actlayer1(x)
        x = self.convlayer2(x)
        if (training):
            x = self.droplayer2(x)
        x = self.actlayer2(x)

        return x

# This is up-convolution or De-convolution block with
# half the feature size of Convolution block
class conv2DTransposeBlock(tf.keras.layers.Layer):

    def __init__(self, filter_size, kernel_size, pool_size, **kwargs):
        super(conv2DTransposeBlock, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.pool_size   = pool_size
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=np.sqrt(2 / (pool_size ** 2 * self.filter_size)))
        self.upconv = tf.keras.layers.Conv2DTranspose(filter_size // 2,
                                                      kernel_size=(pool_size, pool_size),
                                                      kernel_initializer=self.initializer,
                                                      strides=pool_size, padding='valid')

        self.actlayer = tf.keras.layers.Activation('relu')

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.upconv(x)
        x = self.actlayer(x)
        return x

class cropConcat(tf.keras.layers.Layer):

    # def __init__(self, **kwargs):
    #     super(cropConcat, self).__init__(**kwargs)

    def call(self, convLayer, upconvLayer):

        cshape = tf.shape(convLayer)
        upcshape = tf.shape(upconvLayer)

        hstart = (cshape[1] - upcshape[1]) // 2 #to accomodate padding on both sides
        hend = upcshape[1] + hstart
        wstart = (cshape[2] - upcshape[2]) // 2
        wend = upcshape[2] + wstart

        # crop convlayer to upconvlayer output shape [batches, height, width, feature maps]
        croppedLayer = convLayer[:,hstart:hend,wstart:wend,:]

        # Concatenate the feature maps(last dimension) for convLayer and upConvLayer
        x = tf.concat([croppedLayer, upconvLayer], axis=-1)

        return x

# Build the model
inputs = tf.keras.Input(shape=(None, None, 3), name="inputs")
OUTPUT_CLASSES = 3

# Contracting path
iobj = conv2DBlock(64, 3, 0.5)(inputs) # Depth 1
cl1 = iobj
iobj = tf.keras.layers.MaxPooling2D((2, 2))(iobj)

iobj = conv2DBlock(128, 3, 0.5)(iobj)   # Depth 2
cl2  = iobj
iobj = tf.keras.layers.MaxPooling2D((2, 2))(iobj)

iobj = conv2DBlock(256, 3, 0.5)(iobj)   # Depth 3
cl3  = iobj
iobj = tf.keras.layers.MaxPooling2D((2, 2))(iobj)

iobj = conv2DBlock(512, 3, 0.5)(iobj)   # Depth 4
cl4  = iobj
iobj = tf.keras.layers.MaxPooling2D((2, 2))(iobj)

iobj = conv2DBlock(1024, 3, 0.5)(iobj)   # Depth 5

# Expansive path
iobj = conv2DTransposeBlock(1024, 2, 2)(iobj) # Depth 1
iobj = cropConcat()(cl4,iobj)
iobj = conv2DBlock(512, 3, 0.5)(iobj)

iobj = conv2DTransposeBlock(512, 2, 2)(iobj) # Depth 2
iobj = cropConcat()(cl3,iobj)
iobj = conv2DBlock(256, 3, 0.5)(iobj)

iobj = conv2DTransposeBlock(256, 2, 2)(iobj) # Depth 3
iobj = cropConcat()(cl2,iobj)
iobj = conv2DBlock(128, 3, 0.5)(iobj)

iobj = conv2DTransposeBlock(128, 2, 2)(iobj) # Depth 4
iobj = cropConcat()(cl2,iobj)
iobj = conv2DBlock(64, 3, 0.5)(iobj)

iobj = conv2DBlock(64, 3, 0.5)(iobj) # extra to make output 68by68 to 64by64
iobj = conv2DTransposeBlock(128, 2, 2)(iobj)

outputs = tf.keras.layers.Conv2D(filters=OUTPUT_CLASSES,
                      kernel_size=(1, 1),
                      strides=1,
                      padding='valid')(iobj)

#iobj = tf.keras.layers.Activation('relu')(iobj)
#outputs = tf.keras.layers.Activation("softmax", name="outputs")(iobj)
model = tf.keras.Model(inputs, outputs, name="unet")
model.summary()

epochs = 1
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
              metrics = ['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)

def create_mask(pred_mask):
    '''predicted mask is or 3 channels, one for each class
       create the mask having maximum value on the channel axis
       so argmax on last dimension of the predicted mask '''
    print(pred_mask[:, :, 1])
    print(pred_mask[:, :, 2])
    print(pred_mask[:, :, 3])
    pred_mask = tf.argmax(pred_mask, axis= -1)
    # Add the new dimension to the above mask
    pred_mask = pred_mask[..., tf.newaxis]
    print(pred_mask)
    return pred_mask[0] # first image of the batch

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):

        if (logs.get('accuracy') > 0.97):
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True

        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))] )
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

callbacks = DisplayCallback()

# history = model.fit(train_batches,
#                     epochs=epochs,
#                     callbacks=callbacks,
#                     validation_data=test_batches,
#                     steps_per_epoch=STEPS_PER_EPOCH,
#                     validation_steps=VALIDATION_STEPS
#                     )

mpath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\Models\3'
saveModel = os.path.join(mpath,'is_0')
#model.save(saveModel, include_optimizer=False)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
# acc      = history.history[ 'accuracy' ]
# val_acc  = history.history[ 'val_accuracy' ]
# loss     = history.history[ 'loss' ]
# val_loss = history.history['val_loss' ]

# epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
# plt.plot  ( epochs,     acc )
# plt.plot  ( epochs, val_acc )
# plt.title ('Training and validation accuracy')
# plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
# plt.plot  ( epochs,     loss )
# plt.plot  ( epochs, val_loss )
# plt.title ('Training and validation loss'   )
# plt.show()

# Load the saved model
loadModel = tf.saved_model.load(saveModel)
loadModel = loadModel.signatures["serving_default"]

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in loadModel.layers[1:]]

#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = loadModel.input, outputs = successive_outputs)

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(sample_image[tf.newaxis, ...])





