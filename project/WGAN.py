
class WGAN(object):

    def __init__(self, steps=-1, lr=0.0001, silent=True):

        self.GAN = GAN(lr=lr)
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.generator = self.GAN.generator()

        if steps >= 0:
            self.GAN.steps = steps

        self.lastblip = time.clock()

        self.noise_level = 0

        # self.ImagesA = import_images(directory, True)
        self.im = dataGenerator(directory, n_images, suffix=suff, flip=True)
        # (self.im, _), (_, _) = cifar10.load_data()
        # self.im = np.float32(self.im) / 255

        self.silent = silent

        # Train Generator to be in the middle, not all the way at real. Apparently works better??
        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones

        self.enoise = noise(8)
        self.enoiseImage = noiseImage(8)

    def train(self):

        # Train Alternating
        # Get Data

        train_data = [self.im.get_batch(BATCH_SIZE), noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones]
        d_loss = self.DisModel.train_on_batch(train_data, [self.ones, self.nones, self.ones])
        g_loss = self.AdModel.train_on_batch([noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones], self.zeros)

        # Print info
        if self.GAN.steps % 20 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D: " + str(d_loss))
            print("G: " + str(g_loss))
            s = round((time.clock() - self.lastblip) * 1000) / 1000
            print("T: " + str(s) + " sec")
            self.lastblip = time.clock()

            # Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000))
            if self.GAN.steps % 1000 == 0:
                self.evaluate(floor(self.GAN.steps / 1000))

        self.GAN.steps = self.GAN.steps + 1

    def train_dis(self):

        # Get Data
        train_data = [self.im.get_batch(BATCH_SIZE), noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones]

        # Train
        d_loss = self.DisModel.train_on_batch(train_data, [self.ones, self.nones, self.ones])

        return d_loss

    def train_gen(self):

        # Train
        g_loss = self.AdModel.train_on_batch([noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones], self.zeros)

        return g_loss

    def evaluate(self, num=0, trunc=2.0):  # 8x4 images, bottom row is constant

        n = noise(32)
        n2 = noiseImage(32)

        im2 = self.generator.predict([n, n2, np.ones([32, 1])])
        im3 = self.generator.predict([self.enoise, self.enoiseImage, np.ones([8, 1])])

        r12 = np.concatenate(im2[:8], axis=1)
        r22 = np.concatenate(im2[8:16], axis=1)
        r32 = np.concatenate(im2[16:24], axis=1)
        r43 = np.concatenate(im3[:8], axis=1)

        c1 = np.concatenate([r12, r22, r32, r43], axis=0)

        x = Image.fromarray(np.uint8(c1 * 255))

        x.save("Results/i" + str(num) + ".jpg")

    def evaluate2(self, s1, s2, n1, n2, num=0, weight=0.5):

        s = normalize((s2 * weight) + (s1 * (1 - weight)))
        n = (n2 * weight) + (n1 * (1 - weight))

        im2 = self.generator.predict([s, n, np.ones([32, 1])])

        r12 = np.concatenate(im2[:8], axis=1)
        r22 = np.concatenate(im2[8:16], axis=1)
        r32 = np.concatenate(im2[16:24], axis=1)
        r43 = np.concatenate(im2[24:], axis=1)

        c1 = np.concatenate([r12, r22, r32, r43], axis=0)

        x = Image.fromarray(np.uint8(c1 * 255))

        x.save("Results/i" + str(num) + ".jpg")

    def evalTrunc(self, num=0, trunc=1.8):

        n = np.clip(noise(16), -trunc, trunc)
        n2 = noiseImage(16)

        im2 = self.generator.predict([n, n2, np.ones([16, 1])])

        r12 = np.concatenate(im2[:4], axis=1)
        r22 = np.concatenate(im2[4:8], axis=1)
        r32 = np.concatenate(im2[8:12], axis=1)
        r43 = np.concatenate(im2[12:], axis=1)

        c1 = np.concatenate([r12, r22, r32, r43], axis=0)

        x = Image.fromarray(np.uint8(c1 * 255))

        x.save("Results/t" + str(num) + ".jpg")

    def saveModel(self, model, name, num):  # Save a Model
        json = model.to_json()
        with open("Models/" + name + ".json", "w") as json_file:
            json_file.write(json)

        model.save_weights("Models/" + name + "_" + str(num) + ".h5")

    def loadModel(self, name, num):  # Load a Model

        file = open("Models/" + name + ".json", 'r')
        json = file.read()
        file.close()

        mod = model_from_json(json, custom_objects={'AdaInstanceNormalization': AdaInstanceNormalization})
        mod.load_weights("Models/" + name + "_" + str(num) + ".h5")

        return mod

    def save(self, num):  # Save JSON and Weights into /Models/
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)

    def load(self, num):  # Load JSON and Weights from /Models/
        steps1 = self.GAN.steps

        self.GAN = None
        self.GAN = GAN()

        # Load Models
        self.GAN.G = self.loadModel("gen", num)
        self.GAN.D = self.loadModel("dis", num)

        self.GAN.steps = steps1

        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()