import numpy as np
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda

datastuff = ImageDataGenerator(
	rescale=1./255,
	rotation_range=50,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.25,
	zoom_range=0.1,
	channel_shift_range = 20,
	horizontal_flip = True ,
	vertical_flip = True ,
	validation_split = 0.2,
	fill_mode='constant')

train_ds = datastuff.flow_from_directory(
        sys.argv[1],
        subset='training',
        class_mode='binary') 

validation_ds = datastuff.flow_from_directory(
        sys.argv[1],
        subset='validation',
        class_mode='binary') 

base_model = keras.applications.VGG16(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(256, 256, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.


# Freeze the base_model
base_model.trainable = False

LEARNING_RATE =0.0005 

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
	loss='binary_crossentropy',
	optimizer=optimizers.Adam(lr=LEARNING_RATE),
	metrics=['acc'])


NUM_EPOCHS = 20

STEP_SIZE_TRAIN=train_ds.n//train_ds.batch_size
STEP_SIZE_VALID=validation_ds.n//validation_ds.batch_size
result=model.fit(train_ds,
 steps_per_epoch =STEP_SIZE_TRAIN,
 validation_data = validation_ds,
 validation_steps = STEP_SIZE_VALID,
 epochs= NUM_EPOCHS
 )

model.save(sys.argv[2])
