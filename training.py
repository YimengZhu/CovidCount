import tensorflow as tf
from model import CascadeMultiTaskModel
from data import MallDataset

data_folder='/home/prak12-2/mall_dataset'
class_num = 10
batch_size = 1

data = MallDataset(data_folder, class_num)
model = CascadeMultiTaskModel(class_num)

model.compile(optimizer = tf.keras.optimizers.Adam(),
             loss=['mse','mse'])
model.fit(data.train_generatore(batch_size),
                   steps_per_epoch = 10,
                   epochs = 50)

