def hyperplaneFitViaTls(data):
    from numpy.linalg import eig
    import numpy as np
    ### Type your code here

    data = np.asarray(data)
    means = []
    for r in data:
        tem = np.mean(r)
        r -= tem
        means.append(tem)
    covar = data.dot(data.transpose())  # /n ?
    w, vr = eig(covar)


    norm = vr[:, 1]

    if len(means) == 2:
        norm = vr[:, 0]

    d = 0
    for i in range(len(means)):
        d -= norm[i]*means[i]

    ans = []

    for ele in norm:
        ans.append(ele)

    ans.append(d)

    ans = np.asarray(ans)

    if ans[0] < 0:
        ans *= -1

    return ans