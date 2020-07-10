import io

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

from confusionmatrix import plot_to_image, plot_confusion_matrix

def load_and_preprocessing(img_address):
    img = tf.io.read_file(img_address)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img,[192,192])
    img /= 128.0
    img -= 1.0
    img = np.reshape(img, (1,192,192,3))
    return img

class Callbacks():
    def __init__(self,test_address,log_dir,model):
        self.test_img = load_and_preprocessing(test_address)
        self.model = model
        self.file_writer_img = tf.summary.create_file_writer(log_dir+'/img')

    def _log_predicted_image(epoch, logs):
        count_cls, density_map = model.predict(self.test_img)
        density_map_visual = density_map*255.0
        with self.file_writer_img.as_default():
            tf.summary.image("Density Map", density_map_visual, step=epoch)
    
    def log_density_map_callback():
        return tf.keras.callbacks.LambdaCallback(on_epoch_end =
                                                 self._log_predicted_image)
    def _log_confusion_matrix(epoch, logs):
        count_cls_raw, density_map = model.predict(self.test_img)
        count_cls = np.argmax(count_cls_raw, axis=1)

        #calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix()



