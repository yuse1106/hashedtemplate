import numpy as np

def hadamard1(n):
    if n == 1:
        return np.array([[1]])
    else:
        h = hadamard1(n // 2)
        h = np.vstack((np.hstack((h, h)), np.hstack((h, -h))))
        if n == 32:
            return h[:32, :32]
        else:
            return h
        

def RRC(X, B, F_lambda):
    X_T = X.T
    Proj_M = np.linalg.solve(np.dot(X_T, X) + F_lambda * np.eye(X.shape[1]), X_T)
    # X_inv = np.linalg.inv(np.dot(X_T, X))
    # Proj_M = np.dot(X_inv, X_T)

    if np.ndim(X) == 1:
        Y = np.eye(len(B))[B]
    else:
        Y = B

    W = np.dot(Proj_M, Y)
    # labels = np.argmax(X @ W, axis = 1)
    # E = np.sum(np.sum((Y - X @ W) ** 2))+ F_lambda * np.sum(W**2)

    return W
    # return W, labels, E

def hadamard2(X, y, bit, Fmap):
    #np.random.seed(RN)
    #r = np.random.choice(range(bit), size=max(y), replace=False)
    r = np.random.choice(range(bit), size=max(y)+1, replace=False)
    HA = hadamard1(bit)

    b = HA[r, :]
    B = b[y, :]

    Pv = RRC(X, B, Fmap['lambda'])
    #Pt, _, _ = RRC(T, B, Fmap['labmda'])

    F_W = Pv
    #F_T = Pt 
    return F_W

