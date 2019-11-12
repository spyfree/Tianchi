import os
import keras
import time
import subprocess
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,BatchNormalization
from keras.applications import MobileNet,VGG16,InceptionV3,InceptionResNetV2,NASNetLarge
from keras.preprocessing import image
from keras import regularizers
from keras.applications.nasnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants,
                                           signature_def_utils, tag_constants,
                                           utils)

os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
import tensorflow as tf

TRAIN_DIR = os.environ['IMAGE_TRAIN_INPUT_PATH']


os.system('pwd')
os.system('cat fetch_package.py')
#h5_file=subprocess.check_output("find .. -name 'nasnet_large_no_top*' -print", shell=True)
# 加载预训练好的模型,获得其绝对路径
h5_file=subprocess.check_output("find .. -name 'garbage_20191111*' -print", shell=True)


#这是预训练所加的最后的layer，这里仅供参考
#base_model=NASNetLarge(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

#x=base_model.output
#x=GlobalAveragePooling2D()(x)
#x=Dense(512,activation='relu')(x) #dense layer 4
#x=Dropout(0.2)(x) #dropout layer 3
#preds=Dense(100,activation='softmax')(x) #final layer with softmax activation

#model=Model(inputs=base_model.input,outputs=preds)


#加载模型
import keras
model = keras.models.load_model(h5_file.rstrip())


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(331,331),
                                                 color_mode='rgb',
                                                 batch_size=16,
                                                 class_mode='categorical',
                                                 shuffle=True)

#validation_generator=train_datagen.flow_from_directory(TRAIN_DIR,
#                                                 target_size=(299,299),
#                                                 color_mode='rgb',
#                                                 batch_size=32,
#                                                 class_mode='categorical',
#                                                 shuffle=True)

#print(train_generator.class_indices)

step_size_train=train_generator.n//train_generator.batch_size


# 设置577之后的layer是可train
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[577:]:
    layer.trainable = True
    print layer.name

# 设置学习率
model.compile(optimizer=Adam(1e-5),loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=8
                   )

#model.compile(optimizer=Adam(1e-6),loss='categorical_crossentropy',metrics=['accuracy'])
#model.fit_generator(generator=train_generator,
#                   steps_per_epoch=step_size_train,
#                   epochs=2
#                   )


# 这里先使用tf中keras保存h5
export_dir=os.environ['MODEL_INFERENCE_PATH']
tf.keras.models.save_model(model, './test.h5')

time.sleep(10)

# 重新加载保存成savedmodel
save_model = tf.keras.models.load_model('./test.h5')
tf.contrib.saved_model.save_keras_model(save_model, export_dir + '/SavedModel')
