from scipy.io import loadmat
import tensorflow as tf
import numpy as np
import glob
import os
import cv2

def preprocess_image(image):
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image,tf.float32)
    image /= 255.0
    return image
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

class MallDataset():
    def __init__(self, data_folder, class_num):
        image_folder = os.path.join(data_folder, 'frames', '*.jpg')
        self.image_files = sorted(glob.glob(image_folder), key=lambda x: int(x[-8:-4]))

        label_file = os.path.join(data_folder, 'mall_gt.mat')
        annotation = loadmat(label_file)
        self.count = annotation['count']
        self.position = np.expand_dims(annotation['frame'],0)

        max_count, min_count = np.amax(self.count), np.amin(self.count)
        self.class_distance = (max_count - min_count) // class_num

    def _generate_heatmap(self, index, height, width):
        heatmap = np.zeros((height, width))
        for i in self.position[0][0][index][0]:
            for p in i[0]:
                x, y = int(np.round(p[1])),int(np.round(p[0]))
                heatmap[x][y] += 1
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        return heatmap

    def train_generatore(self, batch_size):
        idx_train = list(range(int(len(self.image_files) * 0.8)))

        start_idx = 0
        while True:

            if start_idx + batch_size >= len(idx_train):
                np.random.shuffle(idx_train)
                i = 0
                continue

            batch_input_img, batch_des_img, batch_des_cls = [], [], []
            for idx in range(start_idx, start_idx + batch_size):
                input_img = load_and_preprocess_image(self.image_files[idx])
                des_img = self._generate_heatmap(idx,input_img.shape[0],input_img.shape[1])
                des_cls = self.count[idx] // self.class_distance
                batch_input_img.append(input_img)
                batch_des_img.append(des_img)
                batch_des_cls.append(des_cls)
            start_idx += batch_size
            yield np.array(batch_input_img), [np.array(batch_des_cls),np.array(batch_des_img)]


