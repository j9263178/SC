import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def hist_fit(hist):
    ret = np.array([100., 1., 0.05])

    ### START YOUR CODE HERE ###

    u2 = 0.1
    s2 = 0.4
    u3 = 0.7
    s3 = 0.2
    pi = np.pi

    xmin, xmax, xbinwidth = 0., 2., 0.02
    vx = np.linspace(xmin + xbinwidth / 2, xmax - xbinwidth / 2, 100)
    vy = hist
    vyerr = vy ** 0.5

    def model(x, n1, n2, n3, n4, u1, s1, t):
        # xp = (x - xmin) / (xmax - xmin)
        # polynomial = c0 + c1 * xp + c2 * xp ** 2

        gaussian1 = n1 * xbinwidth / (2. * pi) ** 0.5 / s1 * \
                    np.exp(-0.5 * ((x - u1) / s1) ** 2)
        gaussian2 = n2 * xbinwidth / (2. * pi) ** 0.5 / s2 * \
                    np.exp(-0.5 * ((x - u2) / s2) ** 2)
        gaussian3 = n3 * xbinwidth / (2. * pi) ** 0.5 / s3 * \
                    np.exp(-0.5 * ((x - u3) / s3) ** 2)

        f = n4 * np.exp(-x / t)

        return gaussian1 + gaussian2 + gaussian3 + f

    paras_init = np.array([1000, 300, 200, 200, 1.00, 0.04, 3.0])
    bounds = ((1000,0,0,0,0,0,0),(2000,1000,1000,1000,2,1,100))
    r, c = opt.curve_fit(model, vx, vy, p0=paras_init, sigma=vyerr, maxfev=500000
                         , bounds=bounds)
    ret = np.array([r[0], r[4], r[5]])

    #### END YOUR CODE HERE ####
    return ret

def model(x, n1, n2, n3, n4, u1, s1, t):
        xbinwidth = 0.02
        u2 = 0.1
        s2 = 0.4
        u3 = 0.7
        s3 = 0.2
        pi = np.pi
        gaussian1 = n1 * xbinwidth / (2. * pi) ** 0.5 / s1 * \
                    np.exp(-0.5 * ((x - u1) / s1) ** 2)
        gaussian2 = n2 * xbinwidth / (2. * pi) ** 0.5 / s2 * \
                    np.exp(-0.5 * ((x - u2) / s2) ** 2)
        gaussian3 = n3 * xbinwidth / (2. * pi) ** 0.5 / s3 * \
                    np.exp(-0.5 * ((x - u3) / s3) ** 2)

        f = n4 * np.exp(-x / t)

        return gaussian1 + gaussian2 + gaussian3 + f

if __name__ == '__main__':
    data = np.load('hist_data.npy')

    idx = np.random.randint(50)

    for i in range(0, 50):
        print(i, end=' ')
        ret = hist_fit(data[i])
        xmin, xmax, xbinwidth = 0., 2., 0.02
        vx = np.linspace(xmin + xbinwidth / 2, xmax - xbinwidth / 2, 100)
        vy = data[idx]
        vyerr = vy ** 0.5

        fig = plt.figure(figsize=(6, 6), dpi=80)

        cx = np.linspace(xmin, xmax, 500)
        cy = model(cx, ret[0],ret[1],ret[2],ret[3],ret[4],ret[5],ret[6])
        plt.plot(cx, cy, c='red', lw=2)


        plt.errorbar(vx, vy, vyerr, c='blue', fmt='.')

        print('Signal main Gaussian parameters:')
        print('Area: %.1f' % ret[0])
        print('Mean: %.4f' % ret[4])
        print('Sigma: %.4f' % ret[5])

        plt.grid()
        plt.show()



