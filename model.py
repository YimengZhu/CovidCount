import tensorflow as tf
from tensorflow.keras.layers import Conv2D, PReLu, MaxPooling2D, Flatten, Conv2DTranspose

class CascadeMultiTaskModel(tf.keras.Model):
    def __init__(self, num_class):
        super(CascadeMultiTaskModule, self).__init__(name='')
        self.shared_layer = tf.keras.Sequential()
        self.shared_layer.add(Conv2d(16, 9, padding='same'))
        self.shared_layer.add(PReLu())
        self.shared_layer.add(Conv2d(32, 7, padding='same'))
        self.shared_layer.add(PReLu())

        self.hl_prior1 = tf.keras.Sequential()
        self.hl_prior1.add(Conv2d(16, 9, padding='same'))
        self.hl_prior1.add(PReLu())
        self.hl_prior1.add(MaxPooling2D(2))
        self.hl_prior1.add(Conv2d(32, 7, padding='same'))
        self.hl_prior1.add(PReLu())
        self.hl_prior1.add(MaxPooling2D(2))
        self.hl_prior1.add(Conv2d(16, 7, padding='same'))
        self.hl_prior1.add(PReLu())
        self.hl_prior1.add(Conv2d(8, 7, padding='same'))
        self.hl_prior1.add(PReLu())

        self.hl_prior2 = tf.keras.Sequential()
        self.hl_prior2.add(Conv2d(4, 1, padding='same'))
        self.hl_prior2.add(PReLu())
        self.hl_prior2.add(Flatten())
        self.hl_prior2.add(Dense(512))
        self.hl_prior2.add(Dense(256))
        self.hl_prior2.add(Dense(num_class))

        self.dens1 = tf.keras.Sequential()
        self.dens1.add(Conv2d(20, 7, padding='same'))
        self.dens1.add(PReLu())
        self.dens1.add(MaxPooling2D(2))
        self.dens1.add(Conv2d(40, 5, padding='same'))
        self.dens1.add(PReLu())
        self.dens1.add(MaxPooling2D(2))
        self.dens1.add(Conv2d(20, 5, padding='same'))
        self.dens1.add(PReLu())
        self.dens1.add(Conv2d(10, 5, padding='same'))
        self.dens1.add(PReLu())

        self.dense2 = tf.keras.Sequential()
        self.dense2.add(Conv2d(24, 3, padding='same'))
        self.dense2.add(PReLu())
        self.dense2.add(Conv2d(32, 3, padding='same'))
        self.dense2.add(PReLu())
        self.dense2.add(Conv2DTranspose(16, 4, stride=2))
        self.dense2.add(PReLu())
        self.dense2.add(Conv2DTranspose(8, 4, stride=2))
        self.dense2.add(Conv2d(1, 1, padding='same', activation='relu'))


    def call(self, image):
        shared_feature = slef.shared_layer(image)
        hl_prior1 = self.hl_prior1(shared_feature)
        hl_cls = self.hl_prior2(hl_prior1)
        dense1 = self.dense1(shared_feature)
        dense_map = self.dense2(tf.concat(hl_prior1, dense1), 1)
        return hl_cls, dense_map



