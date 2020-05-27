def myPvAlign(p, q):
    m = len(p)
    n = len(q)

    dpPath = []

    def D(i, j, path):
        if j == 1 and i == 1:
            return abs(p[i - 1] - q[j - 1])
        elif i > 1 and j >= 1:
            px = path.copy()
            py = path.copy()
            x = D(i - 1, j, px)
            y = D(i - 1, j - 1, py)
            if x < y:
                path.append([i-1, j])
                path.extend(px)
                return abs(p[i - 1] - q[j - 1]) + x
            else:
                path.append([i-1, j-1])
                path.extend(py)
                return abs(p[i - 1] - q[j - 1]) + y
        else:
            return 999999

    res = []
    mind = 999999
    for k in range(1, n + 1):
        dpPath.clear()
        cur = D(m, k, dpPath)
        if cur < mind:
            mind = cur
            re =[]
            for i in range(1, len(dpPath)+1):
                re.append(dpPath[-i])
            res = [cur, re]

    return res
import numpy as np
record = -np.ones([4, 6])
print(record)
#qt = [11, 15, 10, 18]
#pt = [12, 11, 15, 16, 11, 12, 9, 10]
#print(myPvAlign(pt, qt))
# https://www.superhentais.com/hentai-anime/mesu-kyoushi-4-kegasareta-kyoudan/2272?fbclid=IwAR1-xQgKhHIQ4d_thDfCll5uRQJhSkSK7Faxv7VCAwrQtGVu6mM0Y9I2hjI
