import numpy as np

def quad_prob():
    Q = np.array([[1.0, 0.0], [0.0, -1.0]])
    A = np.array([0.0,1.0])
    b = np.array([1.0])
    theta_star = np.array([0,1])
    lambda_star = 1

    return Q,A,b,theta_star,lambda_star

def compute_bounds(Q,A,rho):
    P = A.reshape(2,1)
    L_rho = Q + rho*P@P.T
    M = np.max(np.linalg.eig(L_rho)[0])
    m = np.min(np.linalg.eig(L_rho)[0])
    S = np.max(np.linalg.eig(P@P.T)[0])

    eps = 10**(-8)
    b1 = 0.9
    gamma_bound = m**2/(4*S)
    alpha_bound = 2*(b1+1)*np.sqrt(eps)/((1-b1)*M)

    return alpha_bound,gamma_bound