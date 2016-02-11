""" ?
"""
def forward_backward(self):
    X = self.data['X']
    L = self.params['L']
    Tr = self.params['T']
    S = self.params['S']
    G = self.params['G']
    N = X.shape[0]
    T = X.shape[1]

    A = np.array([[1-Tr,Tr],[0,1]])
    B = np.array([[1-G,G],[S,1-S]])
    P = np.array([1-L, L])

    for n in range(N):
        alpha = np.zeros( (T,2) )
        beta = np.zeros( (T,2) )

        beta[T-1] = np.ones(2)
        alpha[0] = np.multiply(P, B[:,X[n,0]])

        for t in range(1,T):
            alpha[t,:] = np.dot( np.multiply( B[:,X[n,t]], alpha[t-1,:]), A)
        for t in range(T-2,0,-1):
            beta[t,0] = np.dot(np.multiply(beta[t+1,:], A[0,:]), B[:, X[n, t+1]])
            beta[t,1] = np.dot(np.multiply(beta[t+1,:], A[1,:]), B[:, X[n, t+1]])
        D = sum(alpha[T-1,:])

        self.params['Y'][n,:] = (np.multiply(alpha, beta) / D)[:,1]
		

