##########################################
# Classification with Image augmentation #
##########################################
# import the packages
import tensorflow as tf
from tensorflow import keras
import os
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from methods import visIntermediate, dispImg

print(tf.__version__)

# Read the images from zip
path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data'
mpath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Saved_Models\Models\3'
folder = 'cats_and_dogs_filtered'
catdog_file = os.path.join(path,folder)

train_dir = os.path.join(catdog_file, 'train')
validation_dir = os.path.join(catdog_file, 'validation')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# see the number of images
print('total training cat images :', len(os.listdir( train_cats_dir ) ))
print('total training dog images :', len(os.listdir( train_dogs_dir ) ))

print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))
print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))

train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

# Peek in to sample images
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index = random.sample(train_cat_fnames, k=8)

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in pic_index
               ]
pic_index = random.sample(train_dog_fnames, k=8)
next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in pic_index
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

# Design the model
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# See the summary of the model
model.summary()

# Use generators to normalize (rescale) and resize
# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
)
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))
# Binary classification - so sigmoid activation for last layer
# and binary cross entropy for loss
model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics = ['accuracy'])

# history = model.fit(train_generator,
#                               validation_data=validation_generator,
#                               steps_per_epoch=100,
#                               epochs=100,
#                               validation_steps=50,
#                               verbose=2)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
# acc      = history.history[ 'accuracy' ]
# val_acc  = history.history[ 'val_accuracy' ]
# loss     = history.history[ 'loss' ]
# val_loss = history.history['val_loss' ]
#
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
#
# saveModel = os.path.join(mpath,'ic1.h5')
# model.save(saveModel)

# sys.exit()

# Load the saved model
loadModel = tf.keras.models.load_model(os.path.join(mpath,'ic1.h5'))

# Predict using the model
fp = os.path.join(path+'/cats_and_dogs_filtered','test')
flist = os.listdir(fp)

for file in flist:
    # predicting images
    path = os.path.join(fp, file)
    dispImg(path)
    img = image.load_img(path, target_size=(150, 150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = loadModel.predict(images, batch_size=10)

    print(classes[0])

    if classes[0] > 0:
        print(file + " is a dog")

    else:
        print(file + " is a cat")

# img = os.path.join(train_dogs_dir,random.choice(train_dog_fnames))
# visIntermediate(loadModel,img)