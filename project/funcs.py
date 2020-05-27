import numpy as np

# Style Z
def noise(n):
    return np.random.normal(0.0, 1.0, size=[n, latent_size])

# Noise Sample
def noiseImage(n):
    return np.random.uniform(0.0, 1.0, size=[n, im_size, im_size, 1])

def getDataSet(path):
    #'/content/drive/My Drive/ml100-03-final/anime01.npz'
    print("Loading dataset...")
    X_train = np.load(path)['arr_0']
    X_train.astype('float32')
    X_train = X_train / 127.5 - 1
    return X_train

def get_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return X, y


def get_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = -ones((n_samples, 1))
    return X, y


# Import Images Function
def import_images(loc, flip=True, suffix='png'):
    out = []
    cont = True
    i = 1
    print("Importing Images...")

    while (cont):
        try:
            temp = Image.open("data/" + loc + "/im (" + str(i) + ")." + suffix + "").convert('RGB')
            temp = temp.resize((im_size, im_size), Image.BICUBIC)
            temp1 = np.array(temp.convert('RGB'), dtype='float32') / 255
            out.append(temp1)
            if flip:
                out.append(np.flip(out[-1], 1))

            i = i + 1
        except:
            cont = False

    print(str(i - 1) + " images imported.")

    return np.array(out)


def normalize(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-7)

# r1/r2 gradient penalty
def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                             axis=np.arange(1, len(gradients_sqr.shape)))
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)