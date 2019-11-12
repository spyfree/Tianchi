# -*- coding: UTF-8 -*-

"""预训练代码，生成garbage_20191111.h5文件"""

import keras
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,Flatten,BatchNormalization,Input
from keras.applications import MobileNet,ResNet50,VGG16,VGG19,InceptionV3,InceptionResNetV2,NASNetLarge
from keras.preprocessing import image
from keras.applications.nasnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam,Nadam
from keras import regularizers 
import tensorflow as tf
import numpy as np
import time

base_model=NASNetLarge(weights='imagenet',include_top=False)

CLASS_NUM=100

x=base_model.output
x=GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x=Dropout(0.2)(x)
preds=Dense(CLASS_NUM,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=preds)

# 所有层均可训练
for layer in model.layers:
    layer.trainable = True

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
                                rotation_range=10,
                                width_shift_range=0.2,
                                height_shift_range=0.1,
                                shear_range=0.1,
                                zoom_range=0.1)

## 实际预训练使用的是colab，这里root_dir可以设置为项目的目录，而TRAIN_DIR是包含了图片的文件夹，由于图片过多这里不提供
#from google.colab import drive
#drive.mount('/content/gdrive', force_remount=True)
#root_dir = "/content/gdrive/My Drive/"
root_dir = '/Users/chenxuyuan/jianguoyun/notes/competition/'
TRAIN_DIR= root_dir + 'total_image/'

train_generator=train_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(331,331), #NASNetLarge
                                                 color_mode='rgb',
                                                 batch_size=8, #防止内存耗尽
                                                 class_mode='categorical',
                                                 shuffle=True)

# 以比较小的速率训练
model.compile(optimizer=Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size

start_time = time.time()
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=10)
print("Model run time: %.2f s"%( (time.time()-start_time)))
model.compile(optimizer=Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
h5_file = root_dir + 'garbage_20191111.h5'
keras.models.save_model(model, h5_file)

