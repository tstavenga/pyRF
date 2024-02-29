import scipy.linalg
import numpy as np

def adjugate(matrix):
    u, sigma, v = scipy.linalg.svd(matrix)

    nullity = len(np.nonzero(sigma))
    
    if nullity >= 2:
        return np.zeros((len(sigma), len(sigma)))
    
    gamma = []
    for i in range(len(sigma)):
        gamma.append(np.prod(sigma[:i])*np.prod(sigma[i+1:]))
    
    adjugate_sigma = np.diag(np.array(gamma))

    adjugate = scipy.linalg.det(v) * scipy.linalg.det(u) * v.conj().transpose() @ adjugate_sigma @ u.conj().transpose()


    return adjugate


if __name__ == '__main__':
    A = np.diag(np.array([-1,2]))
    A = np.arange(1,5).reshape((2,2))
    A = np.ones((2, 2))
    # print(np.linalg.svd(A))
    adjugate(A)