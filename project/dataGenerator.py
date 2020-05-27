
# This is the REAL data generator, which can take images from disk and temporarily use them in your program.
# Probably could/should get optimized at some point
class dataGenerator(object):

    def __init__(self, loc, n, flip=True, suffix='png'):
        self.loc = "data/" + loc
        self.flip = flip
        self.suffix = suffix
        self.n = n

    def get_batch(self, amount):

        idx = np.random.randint(0, self.n - 1, amount) + 1
        out = []

        for i in idx:
            temp = Image.open(self.loc + "/im (" + str(i) + ")." + self.suffix + "").convert('RGB')
            temp1 = np.array(temp.convert('RGB'), dtype='float32') / 255
            if self.flip and random() > 0.5:
                temp1 = np.flip(temp1, 1)

            out.append(temp1)

        return np.array(out)

