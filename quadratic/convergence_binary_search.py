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

start_time = time.time()

gamma_min = gamma_bound
gamma_max = gamma_bound*10**2
alpha_min = alpha_bound*10**(-2)
alpha_max = alpha_bound*10**(4)

boundary_points = []
alpha_step = 10**(1/20)
current_alpha = alpha_max
i = 0
while current_alpha >= alpha_min*(1-10**(-6)):
    print(i)
    start_time_iter = time.time()
    low = gamma_min
    high = gamma_max
    solver_params['steps'] = 100000
    while low < high*0.98:
        mid = np.sqrt(low*high)
        solver_params['lr_primal'] = current_alpha
        solver_params['lr_dual'] = current_alpha*mid
        history_dict = pd_adam_quad_eq_solver(Q,A,b,initial_theta,None,solver_params)
        converged = len(history_dict['epoch']) < solver_params['steps']
        if converged:
            low = mid
        else:
            if len(boundary_points) > 0:
                if mid < boundary_points[-1][1] and solver_params['steps']<1000000:
                    solver_params['steps'] = solver_params['steps'] + 100000
                    continue

            high = mid
    if gamma_min < low < gamma_max:
        boundary_points.append((current_alpha, low))
    current_alpha /= alpha_step
    end_time_iter = time.time()
    elapsed_time_iter = end_time_iter - start_time_iter
    print(f"Time taken: {elapsed_time_iter} seconds")
    i += 1
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")

boundary_points = np.vstack(boundary_points).T
np.savez('boundary_points.npz', array1=boundary_points)