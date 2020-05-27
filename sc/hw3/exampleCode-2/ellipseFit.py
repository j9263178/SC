import numpy as np
from scipy import optimize

def ellipseFit(data):
    ...
    sse = 0
    finalr =[]
    def f(x):
        r1, r2 = sseOfEllipseFit(x, data)
        finalr.clear()
        finalr.append(r1[0])
        finalr.append(r2[0])
        f = 0
        for i in range(0, len(data[0])):
           f += (((data[0][i]-x[0])/r1)**2 + ((data[1][i]-x[1])/r2)**2 - 1)**2
        return f

    center0 = [np.mean(data[0]), np.mean(data[1])]
    center = optimize.fmin(f, x0=center0, disp=False)

    ...

    return center[0], center[1], finalr[0], finalr[1]


def sseOfEllipseFit(center, data):
    a = []
    b = []
    for i in range(0, len(data[0])):
        a.append([(data[0][i] - center[0])**2,(data[1][i] - center[1])**2])
        b.append([1])

    res = np.linalg.lstsq(a, b)

    return (1/res[0][0])**(0.5), (1/res[0][1])**(0.5)