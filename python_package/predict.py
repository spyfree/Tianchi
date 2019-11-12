from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from estimator.create_estimator import Classifier

from nets.vgg16 import VGG16

from nets.model import ClassifyModel

flags = tf.app.flags
tf.flags.DEFINE_string('model_path', '/tmp/train/ResNet50.h5', 'The restored model path')
#tf.flags.DEFINE_string('img_path', '/home/leike/proj/traffic_sign/predict_image/10/9e9485c6bf334e9f8ac275b453b1dc85.jpg', 'The image path')
#tf.flags.DEFINE_string('img_path', '/home/dev/cindy/tensorflow-image-classifier/chongdianbao.jpg', 'The image path')
#tf.flags.DEFINE_string('img_path', '/home/dev/cindy/image_classification/training_dataset/chongdianbao/chongdianbao_017.jpg', 'The image path')
tf.flags.DEFINE_string('img_path', '/home/dev/cindy/image_classification/caiban.jpg', 'The image path')

FLAGS = flags.FLAGS

def input_fn(image_path):
    image_data = tf.gfile.GFile(image_path, 'rb').read()
    decode_image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize_images(decode_image, [224, 224])
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, axis=0)
    return image

def run(flags):
    model_path = flags.model_path
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    input_shape = (224, 224, 3)
    model = tf.keras.models.load_model(model_path)
    inputs = input_fn(flags.img_path)
    result = model.predict(inputs, steps=1)
    result
    print(result)
    return

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    run(FLAGS)

if __name__ == '__main__':
    tf.app.run()
