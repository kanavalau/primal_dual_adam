import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
np.random.seed(0)

def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

X = np.linspace(0, 1, 41).reshape(-1, 1)

l = 0.1
s = 1.0

K = rbf_kernel(X, X,l,s)

K += 1e-10 * np.eye(len(X))

L = cholesky(K, lower=True)
f_prior = np.dot(L, np.random.normal(size=(len(X), 1)))
np.savez('data_gp.npz',array1=X,array2=f_prior)

plt.figure(figsize=(10, 5))
plt.plot(X, f_prior,'x')
plt.title('Sample from a Gaussian Process')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.grid(True)
plt.show()