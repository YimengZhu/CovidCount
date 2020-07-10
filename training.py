import tensorflow as tf
import datetime
from model import CascadeMultiTaskModel
from data import MallDataset

#!rm -rf ./logs/

data_folder= '/tmp/mall_dataset'
train_folder='/tmp/mall_dataset/train'
validation_folder = '/tmp/mall_dataset/validation'
class_num = 10
batch_size = 10

data = MallDataset(data_folder, train_folder, validation_folder, class_num)
model = CascadeMultiTaskModel(class_num)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

model.compile(optimizer = tf.keras.optimizers.Adam(),
             loss=['sparse_categorical_crossentropy','mse'],
             metrics =['mae'])
model.fit(data.train_generator(batch_size,Mode='train'),
          validation_data = data.train_generator(batch_size,Mode='validation'),
                   epochs = 200, callbacks=[tensorboard_callback],
                    steps_per_epoch = 10,
                    validation_steps = 1)

