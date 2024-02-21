import sys
sys.path.insert(0, '../utilities')

from quad_prob_def import quad_prob,compute_bounds
from pd_adam_solver import pd_adam_quad_eq_solver

import numpy as np
import time

Q,A,b,_,_ = quad_prob()

rho = 2.0
alpha_bound,gamma_bound = compute_bounds(Q,A,rho)

initial_theta = np.array([1.5, 1.5])

conv_tol = 10**(-2)
solver_params = {'lr_primal': alpha_bound, 'lr_dual':alpha_bound*gamma_bound, 'rho':rho, 'steps': 150000, 'pretrain_steps':0,'record_steps':1,'plots_folder':'test1','convergence_tolerance':conv_tol}

N = 40

gammas = np.logspace(np.log10(gamma_bound) - 1,np.log10(gamma_bound)+2,N)
alphas = np.logspace(np.log10(alpha_bound) - 2,np.log10(alpha_bound)+4,N)

AS,GS = np.meshgrid(alphas, gammas)

conv = np.zeros_like(AS)
for i in range(N):
    print(i)
    start_time_iter = time.time()
    for j in range(N):
        solver_params['lr_primal'] = AS[i,j]
        solver_params['lr_dual'] = AS[i,j]*GS[i,j]
        history_dict = pd_adam_quad_eq_solver(Q,A,b,initial_theta,None,solver_params)
        conv[i,j] = len(history_dict['epoch']) < solver_params['steps']
    end_time_iter = time.time()
    elapsed_time_iter = end_time_iter - start_time_iter
    print(f"Time taken: {elapsed_time_iter} seconds")

np.savez('convergence_points.npz', array1=AS,array2=GS,array3=conv)