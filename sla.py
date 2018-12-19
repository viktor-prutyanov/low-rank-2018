import numpy as np
from scipy.linalg import sqrtm

def sample_entry(M, delta):
    
    for i in range(M.shape[1]):
        sample = np.random.choice(2, M.shape[0], p=[1 - delta, delta])
        M[:, i] = M[:, i] * sample

    return M


def SPCA(C, k):
    omega = np.random.randn(C.shape[1], k)
    
    print ("gauss random matrix shape ", omega.shape)
    
    _C = np.zeros(C.shape)
    # Trimming
    for i in range (C.shape[0]):
        if (np.count_nonzero(C[i,:]) > 10):
            _C[i,:] = 0
        else:
            _C[i,:] = C[i,:]
    print ("_C shape ", _C.shape)        
    
    F = _C.T @ _C - np.diag(np.diag( _C.T @ _C))
    
    print ("F shape ", F.shape)
    
    QR = (np.linalg.matrix_power(F,
                                 int(np.ceil(5 * np.log(C.shape[1])))) @ omega)
    
    print ("QR shape ", QR.shape)
    
    Q, _ = np.linalg.qr(QR)
    
    return Q[:, :k]


def SLA(M, k, delta, l):
    
    M = M.astype(np.float64)
    
    m, n = M.shape[0], M.shape[1]
    print ("k ", k)
    print ("l ", l)
    print ("M shape ", M.shape)
    
    #l_samples = list(np.random.choice(n, l, replace=True))
    l_samples = np.arange(l)
    
    print ("sample len ", len(l_samples))
        
    A_b1 = sample_entry(M[:, l_samples], delta)
    A_b2 = sample_entry(M[:, l_samples], delta)
    print("A_b1 shape ", A_b1.shape)
    #print(A_b1)
    #print(A_b2)
    Q = SPCA(A_b1, k)
    print("Q shape ", Q.shape)
    
    M = np.delete(M, l_samples, 1)
    
    for i in range(m):
        if (np.count_nonzero(A_b2[i, :]) > 2):
            A_b2[i, :] = 0
            
    for i in range(l):
        if (np.count_nonzero(A_b2[:, i]) > 10 * delta * m):
            A_b2[:, i] = 0
    
    W = A_b2 @ Q
    V = np.zeros((n, k))
    
    print("W shape", W.shape)
    print("V shape", V.shape)
    
    V[:l, :] = (A_b1).T @ W
    
    ## FIXME: ??? what size ???
    I = A_b1 @ V[:l, :]
    print("I shape 1", I.shape)
    
    ### TODO: Need to remove A_b1, A_b2 and Q from RAM
    
    for t in range(l, n):
        A_t = M[:, t - l]
        V[t, :] = A_t.reshape(1, -1) @ W
        I += A_t.reshape(-1, 1) @ V[t, :].reshape(1, -1)
    
    print("I shape 2", I.shape)
        ## TODO: remove A_t from RAM ???

    print("rank A_b1:", np.linalg.matrix_rank(A_b1))
    print("rank A_b2:", np.linalg.matrix_rank(A_b2))
    print("rank W:", np.linalg.matrix_rank(W))
    print("rank Q:", np.linalg.matrix_rank(Q))
    print("rank V:", np.linalg.matrix_rank(V))
    print("rank I:", np.linalg.matrix_rank(I))
    
    R = sqrtm(V.T @ V)
    R = np.linalg.inv(R)
    
    print("R shape", R.shape)
    
    U = (1.0/delta) * (I @ R @ R.T)
    
    ## FIXME: Need to find out what is here in actual algorithm
    M_k = U @ V.T
    print(M_k.shape)
    
    filler = []
    
    for i in range(n-l-1, n-1):
        filler.append(M_k[:, i])
        filler.append(0.5 * (M_k[:, i] + M_k[:, i+1]))
    
    filler = np.array(filler).T
    print(filler.shape)
    print(M_k[:, n-l:].shape)
    
    M_k[:, :n-2*l] = M_k[:, l:n-l]
    M_k[:, n-2*l:] = filler
    
    return M_k 