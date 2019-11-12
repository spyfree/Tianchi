import pandas as pd
import numpy as np
import os
import subprocess
import keras
import time
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.applications import MobileNet,ResNet50,VGG16,InceptionV3,ResNet50V2,InceptionResNetV2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import SGD

import tensorflow as tf
TRAIN_DIR = os.environ['IMAGE_TRAIN_INPUT_PATH']

h5_file=subprocess.check_output("find .. -name 'vgg16_weights_*' -print", shell=True)
os.system('mkdir -p  ~/.keras/models')
os.system('cp ' +  h5_file.rstrip() + ' ~/.keras/models/')


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(4096, activation='relu')(x) #new FC layer, random init
  #x = Dense(1024,activation='relu')(x)
  x = Dropout(0.5)(x)
  x = Dense(2048,activation='relu')(x)
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model

def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:20]:
     layer.trainable = False
  for layer in model.layers[20:]:
     layer.trainable = True
  model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

base_model=VGG16(weights='imagenet',include_top=False)
model = add_new_last_layer(base_model, 100)
setup_to_transfer_learn(model, base_model)

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

train_generator=train_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=10)

setup_to_finetune(model)

model.fit_generator(
    train_generator,
    steps_per_epoch=step_size_train,
    epochs=10)

export_dir=os.environ['MODEL_INFERENCE_PATH']
tf.keras.models.save_model(model, './test.h5')

time.sleep(10)

save_model = tf.keras.models.load_model('./test.h5')
tf.contrib.saved_model.save_keras_model(save_model, export_dir + '/SavedModel')
