import glob
import os
import numpy as np
import tensorflow as tf
import cv2
import random
from scipy.io import loadmat

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 128.0
    image -= 1.0
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_density_map(path):
    density_maps = tf.io.read_file(path)
    density_maps = tf.image.decode_bmp(density_maps, channels=0)
    density_maps = tf.image.resize(density_maps, [192, 192])
    density_maps /= 255.0
    return density_maps


def one_hot_encoder(cls_num):
    one_hot = np.zeros(10)
    one_hot[cls_num] = 1
    return one_hot


class MallDataset():
    def __init__(self, data_folder, class_num):
        image_folder = os.path.join(data_folder, 'frames', '*.jpg')
        density_folder = os.path.join(data_folder, 'den_maps', '*.bmp')
        image_files = sorted(glob.glob(image_folder), key=lambda x: int(x[-8:-4]))
        density_maps = sorted(glob.glob(density_folder), key=lambda x: int(x[-8:-4]))
        self.all_image_paths = list(image_files)
        self.all_density_paths = list(density_maps)
        self.all_image_paths = [str(path) for path in self.all_image_paths]
        self.all_density_paths = [str(path) for path in self.all_density_paths]
        label_file = os.path.join(data_folder, 'mall_gt.mat')
        annotation = loadmat(label_file)
        self.count = annotation['count']
        max_count, min_count = np.amax(self.count), np.amin(self.count)
        self.class_distance = (max_count - min_count) // class_num +1
        self.cls = (self.count - min_count)//self.class_distance # min=13 max=53 0-10
        self.cls = [one_hot_encoder(cls_num) for cls_num in self.cls]
 
    def dataset(self):
        path_image_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        image_ds = path_image_ds.map(load_and_preprocess_image,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)

        path_density_ds = tf.data.Dataset.from_tensor_slices(self.all_density_paths)

        density_ds = path_density_ds.map(load_density_map,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

        cls_ds = tf.data.Dataset.from_tensor_slices(self.cls)

        total_ds = tf.data.Dataset.zip((image_ds, {"output_1":cls_ds,
                                                   "output_2":density_ds}))

        return total_ds

  
