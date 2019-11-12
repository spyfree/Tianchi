import os
import keras
import time
import subprocess
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.applications import MobileNet,ResNet50,VGG16,InceptionV3
from keras.preprocessing import image
#from keras.applications.mobilenet import preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants,
                                           signature_def_utils, tag_constants,
                                           utils)

import tensorflow as tf

TRAIN_DIR = os.environ['IMAGE_TRAIN_INPUT_PATH']
HEIGHT = 224
WIDTH = 224


os.system('pwd')
os.system('cat fetch_package.py')
#os.system('find .. -name "vgg16*" -print')
h5_file=subprocess.check_output("find .. -name 'vgg16_weights_*' -print", shell=True)
os.system('mkdir -p  ~/.keras/models')
os.system('cp ' +  h5_file.rstrip() + ' ~/.keras/models/')
#os.system('cp ../package/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/')
#os.system('cp ../package/keras.json ~/.keras/')

base_model=VGG16(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dropout(0.4)(x) #dropout layer 3
x=Dense(512,activation='relu')(x) #dense layer 4
preds=Dense(100,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

train_generator=train_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

print(train_generator.class_indices)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=22)

export_dir=os.environ['MODEL_INFERENCE_PATH']
tf.keras.models.save_model(model, './test.h5')

time.sleep(10)

save_model = tf.keras.models.load_model('./test.h5')
tf.contrib.saved_model.save_keras_model(save_model, export_dir + '/SavedModel')
