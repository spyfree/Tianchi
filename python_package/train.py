from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os

import tensorflow as tf

from tensorflow.keras import backend as K

from utils import dataset_util
from process_data.create_tf_record import record_exists, get_filenames_and_classes
from process_data.create_tf_record import _RANDOM_SEED, _NUM_SHARDS, _RATE_VAL_ALL
from process_data.create_tf_record import convert_record
from process_data.read_tf_record import train, val

from nets.model import ClassifyModel

flags = tf.app.flags

tf.flags.DEFINE_string('image_dir', os.environ['IMAGE_TRAIN_INPUT_PATH'], 'The image directory')
tf.flags.DEFINE_string('tfrecord_dir', 'record_monster', 'Temporary directory of record file')
tf.flags.DEFINE_string('model_dir', '/tmp/train', 'Saved model directory')
tf.flags.DEFINE_string('export_dir', os.environ['MODEL_INFERENCE_PATH'], 'The export model directory')
tf.flags.DEFINE_string('model_name', 'ResNet50', 'The model name')
tf.flags.DEFINE_integer('batch_size', '16', 'The training dataset batch size')
tf.flags.DEFINE_integer('train_epochs', '10', 'The training epochs')
tf.flags.DEFINE_integer('gpu_nums', '0', 'Training gpu numbers')

FLAGS = flags.FLAGS

def run(flags):
    """
    Create tensorflow record file from image directory
    """
    assert flags.image_dir, '`image_dir ` missing'
    assert flags.tfrecord_dir, '`tfrecord_dir` missing'
    assert flags.model_dir, '`model_dir` missing'
    if not tf.gfile.IsDirectory(flags.tfrecord_dir):
        tf.gfile.MakeDirs(flags.tfrecord_dir)
    if not tf.gfile.IsDirectory(flags.model_dir):
        tf.gfile.MakeDirs(flags.model_dir)
    if not tf.gfile.IsDirectory(flags.export_dir):
        tf.gfile.MakeDirs(flags.export_dir)

    # Define nerve network input shape
    if flags.model_name in ["VGG16",
                           "VGG19",
                           "ResNet50",
                           "ResNet101",
                           "ResNet152",
                           "ResNet50V2",
                           "ResNet101V2",
                           "ResNet152V2",
                           "ResNeXt50",
                           "ResNeXt101",
                           "ResNeXt152",
                           "MobileNet",
                           "MobileNetV2",
                           "DenseNet121",
                           "DenseNet169",
                           "DenseNet201"]:
        input_shape = (224, 224, 3)
    else:
        input_shape = (299, 299, 3)

    image_dir = flags.image_dir
    record_dir = flags.tfrecord_dir
    model_dir = flags.model_dir
    export_dir = flags.export_dir
    model_name = flags.model_name
    batch_size = flags.batch_size
    train_epochs = flags.train_epochs

    photo_filenames, class_names = get_filenames_and_classes(image_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train and test record
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)

    photo_nums = len(photo_filenames)
    validation_nums = int(photo_nums * _RATE_VAL_ALL)

    training_filenames = photo_filenames[validation_nums:]
    validation_filenames = photo_filenames[:validation_nums]

    if record_exists(record_dir):
        tf.logging.info('Record files already exist')
    else:
        # Convert the training and validation record
        convert_record('train', training_filenames, class_names_to_ids, record_dir)
        convert_record('validation', validation_filenames, class_names_to_ids, record_dir)

        # Finally, write the label file
        label_to_class_names = dict(zip(range(len(class_names)), class_names))
        #dataset_util.write_label_file(label_to_class_names, image_dir)

    tf.logging.info("Translate complete")

    """"
    Begin to train a classifier
    """
    data_format = K.image_data_format()

    classify_model = ClassifyModel(input_shape=input_shape, model_name=model_name, classes=len(class_names), data_format=data_format)
    model = classify_model.keras_model()

    if flags.gpu_nums > 1:
        try:
            model = tf.keras.utils.multi_gpu_model(model, gpus=flags.gpu_nums, cpu_relocation=True)
            tf.logging.info("Training using multiple GPUS")
        except:
            tf.logging.info("Training using single GPU")

    model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Set up training input function
    def train_input_fn():
        ds = train(record_dir, input_shape, data_format)
        ds = ds.cache().shuffle(buffer_size=20000).batch(batch_size)
        ds = ds.repeat(train_epochs)
        return ds
    train_dataset = train_input_fn()

    # Callbacks https://tensorflow.google.cn/tutorials/distribute/keras
    filepath = os.path.join(model_dir, "cp-{epoch:04d}.ckpt")
    tf.logging.info(filepath)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, verbose=1, period=1)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=model_dir)
    callbacks = [tensorboard, checkpoint]
    model.fit(x=train_dataset,
            epochs=train_epochs,
            verbose=1,
            steps_per_epoch=(int(len(training_filenames)/batch_size)),
            callbacks=callbacks)

    SAVED_MODEL_PATH = os.path.join(model_dir, model_name+".h5")
    #tf.keras.models.save_model(model, SAVED_MODEL_PATH)
    tf.logging.info(export_dir + '/SavedModel')
    tf.keras.experimental.export_saved_model(model, export_dir + '/SavedModel')
    tf.logging.info('saved models complete')
    return

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    run(FLAGS)

if __name__ == '__main__':
    tf.app.run()
