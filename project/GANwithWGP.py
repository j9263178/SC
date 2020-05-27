

#================== Config Stuff=================================================
TRAIN_FROM_SCRATCH = False
D_MODEL_PATH = "models/stgcont_80000.h5"
G_MODEL_PATH = "models/stgcont_80000.h5"

LEARNING_RATE = 0.0002
IM_SIZE = 64
LATENT_SIZE = 128
BATCH_SIZE = 64
ENABLE_NOISE = True

IMG_SAVE_INTERVAL = 1000
MODEL_SAVE_INTERVAL = 40000

PATH = "dataset/anime01.npz"
IMG_SAVE_PATH = "fuck/anime_%d.png"
MODEL_SAVE_PATH_G = 'models/stgcont_%d.h5'
MODEL_SAVE_PATH_D = 'models/stdcont_%d.h5'
#================== Config Stuff=================================================

# Imports for layers and models
from keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from keras.models import Model
from keras import models
from keras.optimizers import Adam
import keras.backend as K
from keras.layers import Layer

import shutil
import os

import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial

# Input b and g should be 1x1xC
class AdaInstanceNormalization(Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):

        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')

        super(AdaInstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))

        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):

        return input_shape[0]

def get_noise(n):
    return np.random.normal(0.0, 1.0, size=[n, LATENT_SIZE])

def get_noiseImage(n):
    return np.random.uniform(0.0, 1.0, size=[n, IM_SIZE, IM_SIZE, 1])

# Upsample, Convolution, AdaIN, Noise, Activation, Convolution, AdaIN, Noise, Activation
def g_block(inp, style, noise, fil, u=True):
    # Looks like CIN, Modify here if neccasary
    Beta = Dense(fil)(style)
    Beta = Reshape([1, 1, fil])(Beta)
    Gamma = Dense(fil)(style)
    Gamma = Reshape([1, 1, fil])(Gamma)

    n = Conv2D(filters=fil, kernel_size=1, padding='same', kernel_initializer='he_normal')(noise)

    if u:
        out = UpSampling2D(interpolation='bilinear')(inp)
        out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
    else:
        out = Activation('linear')(inp)

    out = AdaInstanceNormalization()([out, Beta, Gamma])
    out = add([out, n])
    out = LeakyReLU(0.01)(out)

    Beta = Dense(fil)(style)
    Beta = Reshape([1, 1, fil])(Beta)
    Gamma = Dense(fil)(style)
    Gamma = Reshape([1, 1, fil])(Gamma)

    out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
    out = AdaInstanceNormalization()([out, Beta, Gamma])

    if ENABLE_NOISE:
        n = Conv2D(filters=fil, kernel_size=1, padding='same', kernel_initializer='he_normal')(noise)
        out = add([out, n])

    out = LeakyReLU(0.01)(out)

    return out

# Convolution, Activation, Pooling, Convolution, Activation
def d_block(inp, fil, p=True):
    route2 = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(inp)
    route2 = LeakyReLU(0.01)(route2)
    if p:
        route2 = AveragePooling2D()(route2)
    route2 = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(route2)
    out = LeakyReLU(0.01)(route2)

    return out

def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                             axis=np.arange(1, len(gradients_sqr.shape)))
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)

def getDataSet(path):
    print("Loading dataset...")
    X_train = np.load(path)['arr_0']
    X_train.astype('float32')
    X_train = X_train / 255
    print("Done! Get images : " + str(X_train.shape[0]))
    return X_train

class GAN(object):

    def __init__(self):

        self.steps = 1
        self.lastblip = 0

        if TRAIN_FROM_SCRATCH:
            # Two raw models
            self.Discriminator = self.build_discriminator()
            self.Generator = self.build_generator()
        else:
            print("Loading two models...")
            self.Discriminator = models.load_model(D_MODEL_PATH)
            self.Generator = models.load_model(G_MODEL_PATH)
            print("Done!")

        # Two models for training
        self.Combination_Model = self.build_Combination_Model()
        self.Dis_Model = self.build_Dis_Model()

        # Constants
        self.Dataset = getDataSet(PATH)
        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones

    def build_discriminator(self):

        input_from_generator = Input(shape=[IM_SIZE, IM_SIZE, 3])

        # Size
        x = d_block(input_from_generator, 16)  # Size / 2
        x = d_block(x, 32)  # Size / 4
        x = d_block(x, 64)  # Size / 8

        if IM_SIZE > 32:
            x = d_block(x, 128)  # Size / 16

        if IM_SIZE > 64:
            x = d_block(x, 192)  # Size / 32

        if IM_SIZE > 128:
            x = d_block(x, 256)  # Size / 64

        if IM_SIZE > 256:
            x = d_block(x, 384)  # Size / 128

        if IM_SIZE > 512:
            x = d_block(x, 512)  # Size / 256

        x = Flatten()(x)

        x = Dense(128)(x)
        x = Activation('relu')(x)

        x = Dropout(0.6)(x)
        x = Dense(1)(x)

        return Model(inputs=input_from_generator, outputs=x)

    def build_generator(self):

        # Style FC, I only used 2 fully connected layers instead of 8 for faster training
        latent_input = Input(shape=[LATENT_SIZE])
        sty = Dense(512, kernel_initializer='he_normal')(latent_input)
        sty = LeakyReLU(0.1)(sty)
        sty = Dense(512, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.1)(sty)

        # Get the noise image and crop for each size
        noise_input = Input(shape=[IM_SIZE, IM_SIZE, 1])
        noi = [Activation('linear')(noise_input)]
        curr_size = IM_SIZE
        while curr_size > 4:
            curr_size = int(curr_size / 2)
            noi.append(Cropping2D(int(curr_size / 2))(noi[-1]))

        # Here do the actual generation stuff
        constant_input = Input(shape=[1])
        x = Dense(4 * 4 * 512, kernel_initializer='he_normal')(constant_input)
        x = Reshape([4, 4, 512])(x)
        x = g_block(x, sty, noi[-1], 512, u=False)

        if IM_SIZE >= 1024:
            x = g_block(x, sty, noi[7], 512)  # Size / 64
        if IM_SIZE >= 512:
            x = g_block(x, sty, noi[6], 384)  # Size / 64
        if IM_SIZE >= 256:
            x = g_block(x, sty, noi[5], 256)  # Size / 32
        if IM_SIZE >= 128:
            x = g_block(x, sty, noi[4], 192)  # Size / 16
        if IM_SIZE >= 64:
            x = g_block(x, sty, noi[3], 128)  # Size / 8

        x = g_block(x, sty, noi[2], 64)  # Size / 4
        x = g_block(x, sty, noi[1], 32)  # Size / 2
        x = g_block(x, sty, noi[0], 16)  # Size

        x = Conv2D(filters=3, kernel_size=1, padding='same', activation='sigmoid')(x)

        return Model(inputs=[latent_input, noise_input, constant_input], outputs=x)

    def build_Combination_Model(self):

        self.Discriminator.trainable = False
        self.Generator.trainable = True

        for layer in self.Discriminator.layers:
            layer.trainable = False
        for layer in self.Generator.layers:
            layer.trainable = True

        latent_input = Input(shape=[LATENT_SIZE])
        noise_input = Input(shape=[IM_SIZE, IM_SIZE, 1])
        constant_input = Input(shape=[1])

        gout = self.Generator([latent_input, noise_input, constant_input])
        dout = self.Discriminator(gout)

        Combination_Model = Model(inputs=[latent_input, noise_input, constant_input], outputs=dout)

        Combination_Model.compile(optimizer=Adam(LEARNING_RATE, beta_1=0, beta_2=0.99, decay=0.00001), loss='mse')

        return Combination_Model

    def build_Dis_Model(self):

        self.Discriminator.trainable = True
        self.Generator.trainable = False

        for layer in self.Discriminator.layers:
            layer.trainable = True
        for layer in self.Generator.layers:
            layer.trainable = False

        # Real Pipeline
        real_input = Input(shape=[IM_SIZE, IM_SIZE, 3])
        dreal = self.Discriminator(real_input)

        # Fake Pipeline
        latent_input = Input(shape=[LATENT_SIZE])
        noise_input = Input(shape=[IM_SIZE, IM_SIZE, 1])
        constant_input = Input(shape=[1])
        gout = self.Generator([latent_input, noise_input, constant_input])
        dfake = self.Discriminator(gout)

        # Samples for gradient penalty
        # For r1 use real samples (ri)
        # For r2 use fake samples (gf)
        # Model With Inputs and Outputs
        dreal2 = self.Discriminator(real_input)
        dm = Model(inputs=[real_input, latent_input, noise_input, constant_input], outputs=[dreal, dfake, dreal2])

        # Create partial of gradient penalty loss
        # For r1, averaged_samples = ri
        # For r2, averaged_samples = gf
        # Weight of 10 typically works
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=real_input, weight=5)
        partial_gp_loss.__name__ = 'gradient_penalty'
        # Compile With Corresponding Loss Functions
        dm.compile(optimizer=Adam(LEARNING_RATE, beta_1=0, beta_2=0.99, decay=0.00001),
                   loss=['mse', 'mse', partial_gp_loss])

        return dm

    def train(self):

        # select random samples from datasets
        real_img = self.Dataset[np.random.randint(0, self.Dataset.shape[0], BATCH_SIZE)]
        latent_vector = get_noise(BATCH_SIZE)

        if ENABLE_NOISE:
            noise_image = get_noiseImage(BATCH_SIZE)
        else:
            noise_image = np.zeros([BATCH_SIZE, IM_SIZE, IM_SIZE, 1])

        constant_input = self.ones

        d_loss = self.Dis_Model.train_on_batch(
            [real_img, latent_vector, noise_image, constant_input],
            [self.ones, self.nones, self.zeros]
        )

        g_loss = self.Combination_Model.train_on_batch(
            [latent_vector, noise_image, constant_input],
            self.ones-0.5
        )

        if self.steps % IMG_SAVE_INTERVAL == 0:
            print("\nRound " + str(self.steps) + ":")
            print("D loss: " + str(d_loss))
            print("G loss: " + str(g_loss))
            print(str(round((time.process_time() - self.lastblip) * 1000) / 1000) + " sec")
            self.lastblip = time.process_time()

            if self.steps % IMG_SAVE_INTERVAL == 0:
                self.evaluate()
            if self.steps % MODEL_SAVE_INTERVAL == 0:
                self.saveModel()

        self.steps = self.steps + 1

    def evaluate(self):
        r, c = 5, 5

        n = get_noise(25)
        n2 = get_noiseImage(25)
        gen_imgs = self.Generator.predict([n, n2, np.ones([25, 1])])
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = gen_imgs[:, :, :, ::-1]

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig(IMG_SAVE_PATH % self.steps)

    def saveModel(self):
        self.Discriminator.save(MODEL_SAVE_PATH_D % self.steps)
        self.Generator.save(MODEL_SAVE_PATH_G % self.steps)

def configGPU():
    import tensorflow as tf
    # 自動增長 GPU 記憶體用量
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    # 設定 Keras 使用的 Session
    tf.compat.v1.keras.backend.set_session(sess)

if __name__ == "__main__":

    configGPU()

    if os.path.exists('fuck') and os.path.isdir('fuck'):
        shutil.rmtree('fuck')
    os.makedirs('fuck')

    model = GAN()
    while True:
        model.train()
