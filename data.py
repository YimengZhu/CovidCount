from scipy.io import loadmat
import numpy as np
import cv2
import glob
import os

class MallDataset():
    def __init__(self, data_folder, class_num):
        image_folder = os.path.join(data_folder, 'frames', '*.jpg')
        self.image_files = sorted(glob.glob(image_folder), key=lambda x: int(x[-8:-4]))

        label_file = os.path.join(data_folder, 'mall_gt.mat')
        annotation = loadmat(label_file)
        self.count = annotation['count']
        self.position = annotation['frame'].unsqueeze(0)

        max_count, min_count = numpy.amax(self.count), numpy.amin(self.count)
        self.class_distance = (max_count - min_count) // class_num


    def _generate_heatmap(self, index, height, weight):
        heatmap = np.zeros((height, weight))
        for i in self.position[index][0][0][0]:
            x, y = p[0], p[1]
            heatmap[x,y] += 1
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
                input_img = cv.imread(self.image_files[idx])
                des_img = _generate_heatmap(idx, input_img.size[:-1])
                des_cls = count[idx] // self.class_distance
                batch_input_img.append(input_img)
                batch_des_img.append(des_img)
                batch_des_cls.append(des_cls)
            start_idx += batch_size
            yield np.array(batch_input_img), np.array(batch_des_img), np.array(batch_des_cls)


