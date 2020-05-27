from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
import time

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


class DCGAN():
    def __init__(self):
        self.lastblip = 0
        # Input shape
        self.img_rows = 64
        self.img_cols = self.img_rows
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

       # self.discriminator = self.build_discriminator()

        self.discriminator = models.load_model('models/d64_40000.h5')


        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

       # self.generator = self.build_generator()

        self.generator = models.load_model('models/g64_40000.h5')

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        nodes = int(self.img_rows / 8)
        model = Sequential()

        model.add(Dense(512 * nodes * nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((nodes, nodes, 512)))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        #   model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        #  model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # model.add(Dense(512, activation="tanh"))
        # model.add(Dropout(0.25))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        #  model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.1))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        #   model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        #  model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        #  model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))
        model.add(Flatten())

        model.add(Dense(512, activation="tanh"))
        model.add(Dropout(0.35))
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        print("Loading dataset...")
        X_train = np.load("dataset/anime01.npz")['arr_0']

        # Rescale -1 to 1
        X_train.astype('float32')
        X_train = X_train / 127.5 - 1

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # valid = np.random.uniform(low=0.1, high=0.5 ,size = (batch_size, 1))
        # fake = np.random.uniform(low=0, high=0.3 ,size = (batch_size, 1))
        max = 0

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            noise2 = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise2, valid)

            # If at save interval => save generated image samples
            # Plot the progress

            if epoch % save_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

                print("save!")
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        s = round((time.process_time() - self.lastblip) * 1000) / 1000
        print(str(s) + " sec")
        self.lastblip = time.process_time()

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = gen_imgs[:, :, :, ::-1]

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("fuck/anime_%d.png" % epoch)
        dcgan.discriminator.save('models/d64cont_%d.h5' % epoch)
        dcgan.generator.save('models/g64cont_%d.h5' % epoch)


if __name__ == '__main__':
    # 自動增長 GPU 記憶體用量
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    # 設定 Keras 使用的 Session
    tf.compat.v1.keras.backend.set_session(sess)

    if os.path.exists('fuck') and os.path.isdir('fuck'):
        shutil.rmtree('fuck')
    os.makedirs('fuck')

    dcgan = DCGAN()
    dcgan.train(epochs=100000, batch_size=64, save_interval=1000)