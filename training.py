import shutil
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from model import CascadeMultiTaskModel
from loader_dataset import MallDataset

shutil.rmtree('logs', ignore_errors=True)
# Define callbacks for Tensorboard visualization

log_dir = "logs/fit"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

test_img = '/tmp/mall_dataset/cropped_images/8000.jpg'
file_writer_img = tf.summary.create_file_writer(log_dir + '/img')

def log_predicted_image(epoch, logs):
    test_image = tf.io.read_file(test_img)
    test_image = tf.image.decode_jpeg(test_image, channels=3)
    test_image = tf.image.resize(test_image,[192,192])
    test_image /= 128.0
    test_image -= 1.0
    test_image = np.reshape(test_image,(1,192,192,3))
    count, density_map = model.predict(test_image)
    density_map_visual = density_map*255.0
    with file_writer_img.as_default():
        tf.summary.image("Density Map", density_map_visual, step=epoch)

den_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end =
                                              log_predicted_image)

# Define data location

data_folder='/tmp/mall_dataset'
class_num = 10
batch_size = 16

# Define training/validation dataset
data = MallDataset(data_folder, class_num)
total_ds = data.dataset()

train_ds = total_ds.take(7000)
train_ds = train_ds.shuffle(buffer_size=7000)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)


validation_ds = total_ds.skip(7000)
validation_ds = validation_ds.shuffle(buffer_size=1000)
validation_ds = validation_ds.repeat()
validation_ds = validation_ds.batch(batch_size)
validation_ds = validation_ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

# Training pipeline
model = CascadeMultiTaskModel(class_num)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss={"output_1":'categorical_crossentropy', 
                    "output_2":'mse'},
              loss_weights={"output_1":0.0001, "output_2":1},
              metrics=['mae'])
model.fit(x=train_ds, validation_data=validation_ds,
          steps_per_epoch=875,callbacks=[tensorboard_callback, den_callback], 
           validation_steps=50, epochs = 300)
