import tensorflow as tf
from tensorflow.keras.layers import Conv2D, PReLU, MaxPooling2D, Flatten, Conv2DTranspose, Dense

class CascadeMultiTaskModel(tf.keras.Model):
    def __init__(self, num_class):
        super(CascadeMultiTaskModel, self).__init__(name='')
        self.shared_layer = tf.keras.Sequential()
        self.shared_layer.add(Conv2D(16, 9, padding='same'))
        self.shared_layer.add(PReLU())
        self.shared_layer.add(Conv2D(32, 7, padding='same'))
        self.shared_layer.add(PReLU())

        self.hl_prior1 = tf.keras.Sequential()
        self.hl_prior1.add(Conv2D(16, 9, padding='same'))
        self.hl_prior1.add(PReLU())
        self.hl_prior1.add(MaxPooling2D(2))
        self.hl_prior1.add(Conv2D(32, 7, padding='same'))
        self.hl_prior1.add(PReLU())
        self.hl_prior1.add(MaxPooling2D(2))
        self.hl_prior1.add(Conv2D(16, 7, padding='same'))
        self.hl_prior1.add(PReLU())
        self.hl_prior1.add(Conv2D(8, 7, padding='same'))
        self.hl_prior1.add(PReLU())

        self.hl_prior2 = tf.keras.Sequential()
        self.hl_prior2.add(Conv2D(4, 1, padding='same'))
        self.hl_prior2.add(PReLU())
        self.hl_prior2.add(Flatten())
        self.hl_prior2.add(Dense(512))
        self.hl_prior2.add(PReLU())
        self.hl_prior2.add(Dense(256))
        self.hl_prior2.add(PReLU())
        self.hl_prior2.add(Dense(num_class,activation='sigmoid'))

        self.dense1 = tf.keras.Sequential()
        self.dense1.add(Conv2D(20, 7, padding='same'))
        self.dense1.add(PReLU())
        self.dense1.add(MaxPooling2D(2))
        self.dense1.add(Conv2D(40, 5, padding='same'))
        self.dense1.add(PReLU())
        self.dense1.add(MaxPooling2D(2))
        self.dense1.add(Conv2D(20, 5, padding='same'))
        self.dense1.add(PReLU())
        self.dense1.add(Conv2D(10, 5, padding='same'))
        self.dense1.add(PReLU())

        self.dense2 = tf.keras.Sequential()
        self.dense2.add(Conv2D(24, 3, padding='same'))
        self.dense2.add(PReLU())
        self.dense2.add(Conv2D(32, 3, padding='same'))
        self.dense2.add(PReLU())
        self.dense2.add(Conv2DTranspose(16, 4, strides=2, padding='same'))
        self.dense2.add(PReLU())
        self.dense2.add(Conv2DTranspose(8, 4, strides=2, padding='same'))
        self.dense2.add(Conv2D(1, 1, padding='same', activation='tanh'))
        

    def call(self, image):
        shared_feature = self.shared_layer(image)
        hl_prior1 = self.hl_prior1(shared_feature)
        hl_cls = self.hl_prior2(hl_prior1)
        dense1 = self.dense1(shared_feature)
        dense_map = self.dense2(tf.concat([hl_prior1, dense1],3))
        return hl_cls, dense_map



