import sys
sys.path.insert(0, '../utilities')

from quad_prob_def import quad_prob,compute_bounds
from pd_adam_solver import pd_adam_quad_eq_solver
from plot_utils import plot_contours_and_feasibility
import numpy as np

Q,A,b,_,_ = quad_prob()

rho = 2.0
alpha_bound,gamma_bound = compute_bounds(Q,A,rho)

initial_theta = np.array([1.5, 1.5])

conv_tol = 10**(-2)
solver_params = {'lr_primal': alpha_bound, 'lr_dual':alpha_bound*gamma_bound, 'rho':rho, 'steps': 300000, 'pretrain_steps':0,'record_steps':1,'plots_folder':'test1','convergence_tolerance':conv_tol}

history_dicts = []

solver_params['lr_primal'] = alpha_bound
solver_params['lr_dual'] = gamma_bound*alpha_bound/2.5
history_dicts.append(pd_adam_quad_eq_solver(Q,A,b,initial_theta,None,solver_params))

solver_params['lr_primal'] = alpha_bound
solver_params['lr_dual'] = gamma_bound*alpha_bound*4
history_dicts.append(pd_adam_quad_eq_solver(Q,A,b,initial_theta,None,solver_params))

plot_contours_and_feasibility(Q,A,b,history_dicts)