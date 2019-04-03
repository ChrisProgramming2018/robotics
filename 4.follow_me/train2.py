import os
import sys
import tensorflow as tf

from scipy import misc
import numpy as np



from tensorflow import image
from tensorflow.python import keras
from tensorflow.contrib.keras import layers , models
from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools 
from utils import model_tools
sys.path.append('/usr/lib/python2.7/')
from glob import glob

from keras import backend as k
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.98

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from model1 import separable_conv2d_batchnorm, conv2d_batchnorm, bilinear_upsample
from model1 import encoder_block,decoder_block, fcn_model

print(keras.__version__)
print(tf.__version__)

image_hw = 160
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(shape=image_shape)
num_classes = 3

# Call fcn_model()
output_layer = fcn_model(inputs, num_classes)


learning_rate = 0.004
batch_size = 128
num_epochs = 100
steps_per_epoch = 32
validation_steps = 50
workers = 8

model = models.Model(inputs=inputs, outputs=output_layer)

model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

# Data iterators for loading the training and validation data
train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                               data_folder=os.path.join('..', 'data', 'train'),
                                               image_shape=image_shape,
                                               shift_aug=True)

val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                             data_folder=os.path.join('..', 'data', 'validation'),
                                             image_shape=image_shape)

logger_cb = plotting_tools.LoggerPlotter()
callbacks = [logger_cb]

model.fit_generator(train_iter,
                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                    epochs = num_epochs, # the number of epochs to train for,
                    validation_data = val_iter, # validation iterator
                    validation_steps = validation_steps, # the number of batches to validate on
                    callbacks=callbacks,
                    workers = workers)

