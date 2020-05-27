def myPvAlign(p, q):
    import numpy as np
    m = len(p)
    n = len(q)

    record = -np.ones([m+1, n+1])

    def D(i, j):
        if j == 1 and i == 1:
            return abs(p[i - 1] - q[j - 1])
        elif i > 1 and j >= 1:
            if record[i-1][j] == -1:
                record[i - 1][j] = D(i - 1, j)
            x = record[i - 1][j]

            if record[i - 1][j-1] == -1:
                record[i - 1][j-1] = D(i - 1, j - 1)
            y = record[i - 1][j-1]

            if x < y:
                return abs(p[i - 1] - q[j - 1]) + x
            else:
                return abs(p[i - 1] - q[j - 1]) + y
        else:
            return 999999

    res = 999999
    for k in range(1, n + 1):
        cur = D(m, k)
        if cur < res:
            res = cur

    return res


#qt = [11, 15, 10, 18]
#pt = [12, 11, 15, 16, 11, 12, 9, 10]
#print(myPvAlign(pt, qt))
# https://www.superhentais.com/hentai-anime/mesu-kyoushi-4-kegasareta-kyoudan/2272?fbclid=IwAR1-xQgKhHIQ4d_thDfCll5uRQJhSkSK7Faxv7VCAwrQtGVu6mM0Y9I2hjI
