
def addAndMax(A, B):
    import numpy as np
    A = np.asarray(A)
    B = np.asarray(B)
    outerc = max(A.shape[0], B.shape[0])
    outerr = max(A.shape[1], B.shape[1])

    A = np.pad(A, ((0, outerc - A.shape[0]), (0, outerr - A.shape[1])), mode='constant', constant_values=0)
    B = np.pad(B, ((0, outerc - B.shape[0]), (0, outerr - B.shape[1])), mode='constant', constant_values=0)
    C = A + B
    return C.max()
