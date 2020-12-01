# X = classic_mds(D, dim)
#
# Reconstructs the point set from a noisy distance matrix.
#
# INPUT:   D   ... measured Euclidean Distance Matrix (EDM) (n by n)
#          dim ... target embedding dimension
#
# OUTPUT:  X   ... (dim by n) list of point coordinates
#
#Author: Miguel Muñoz, 2020

import numpy as np
from matplotlib import pyplot as plt

# The implementation of the Classical MDS is performed with Eigenvalue Descomposition rather than SVD 
# The results might differ from the previous implementation.

def classicalMDS(D, dim):

    
    n = D.shape[0]
    I = np.eye(n)
    J = I - 1/n * np.ones((n, n))
    
    G = -0.5*np.matmul(J, np.matmul(D,J))

    l, U = np.linalg.eig(G)
    
    idx = np.argsort(-l)
    l = l[idx]
    U = U[:,idx]    

    
    aux = np.concatenate((np.diag(np.sqrt(l[0:dim])),np.zeros((dim,n-dim))),axis=1)

    return np.matmul(aux,np.transpose(U))

# Example of a point data set
X = np.array([[0 , 1 , 2 , 3 , 4], 
              [0, 1, 0 , 1 , 0]])
# Calculation of the EDM corresponding to X
dim = X.shape[0]
n = X.shape[1]
Gramm = np.matmul(np.transpose(X),X)
diag = np.zeros((n,1))
one = np.ones((n,1))
for i in range(0,len(diag)):
    diag[i] = Gramm[i][i]
    
D = np.outer(diag,one) - 2*Gramm + np.outer(one, diag)

 
# Classical MDS call
res = classicalMDS(D, dim)
print(res)
plt.plot(res[0,:], res[1,:])
