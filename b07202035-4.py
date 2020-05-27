def hist_fit(hist):
    import numpy as np
    import scipy.optimize as opt
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
    bounds = ((1000, 0, 0, 0, 0, 0, 0), (2000, 1000, 1000, 1000, 2, 1, 100))
    r, c = opt.curve_fit(model, vx, vy, p0=paras_init, sigma=vyerr, maxfev=500000
                         , bounds=bounds)

    ret = np.array([r[0], r[4], r[5]])

    #### END YOUR CODE HERE ####

    return ret