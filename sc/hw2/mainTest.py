import numpy as np

def CircleFitting():

    def Load_Coordinates():
        data = [[float(val) for val in input().split()] for i in range(2)]
        return data

    for _ in range(5):
        data = Load_Coordinates()
        a, b, r = circleFitByDss(data)
        print('{} {} {}'.format(a, b, r))


def circleFitByDss(data):

    def func(x):
        f = 0
        for i in range(0, len(data[0])):
            f += abs(((x[0] - data[0][i]) ** 2 + (x[1] - data[1][i]) ** 2) ** (1/2) - x[2])
        return f

    import statistics
    from scipy import optimize

    px = statistics.mean(data[0])
    py = statistics.mean(data[1])
    pr = ((data[0][0]-px)**2+(data[1][0]-py)**2)**(0.5)
    res = optimize.fmin(func, x0=(px, py, pr), disp=False)
    return res[0], res[1], res[2]

if __name__ == '__main__':
    CircleFitting()