import sys
sys.path.insert(0, '../utilities')

from quad_prob_def import quad_prob
from pd_adam_solver import pd_adam_quad_eq_solver
from plot_utils import plot_history
import numpy as np

Q,A,b,_,_ = quad_prob()
initial_theta = np.array([1.5, 1.5])

rho = 2.0

P = A.reshape(2,1)
L_rho = Q + rho*P@P.T
M = np.max(np.linalg.eig(L_rho)[0])
m = np.min(np.linalg.eig(L_rho)[0])
S = np.max(np.linalg.eig(P@P.T)[0])

eps = 10**(-8)
b1 = 0.9
gamma_bound = m**2/(4*S)
alpha_bound = 2*(b1+1)*np.sqrt(eps)/((1-b1)*M)
conv_tol = 10**(-2)
solver_params = {'lr_primal': alpha_bound, 'lr_dual':alpha_bound*gamma_bound, 'rho':rho, 'steps': 100000, 'pretrain_steps':0,'record_steps':1,'plots_folder':'test1','convergence_tolerance':10**(-2)}

history_dict = pd_adam_quad_eq_solver(Q,A,b,initial_theta,None,solver_params)
plot_history(history_dict)