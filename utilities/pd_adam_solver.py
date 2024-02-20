import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from aug_lag_def import Augmented_Lagrangian,Quad_Objective,LinEq_Constraint
from adam import Adam

def gd_step(aug_lag, optimizer_primal):
    # Performs primal-dual iteration step
    optimizer_primal.zero_grad()

    loss = aug_lag.objective()
    loss.backward()
    
    optimizer_primal.step()

def pd_step(aug_lag, optimizer_primal, optimizer_dual):
    # Performs primal-dual iteration step
    optimizer_primal.zero_grad()
    optimizer_dual.zero_grad()

    loss = aug_lag()
    loss.backward()
    
    optimizer_primal.step()
    optimizer_dual.step()

def record_history(hist_dict,step,aug_lag):
    # Appends the data from the most recent iteration to a dictionary
    with torch.no_grad():
        hist_dict['epoch'].append(step)
        hist_dict['objective'].append(aug_lag.objective(aug_lag.primal_var).item())
        if torch.is_tensor(aug_lag.constraint(aug_lag.primal_var)):
            hist_dict['constraint'].append(aug_lag.constraint(aug_lag.primal_var).numpy().copy())
        else:
            hist_dict['constraint'].append(aug_lag.constraint(aug_lag.primal_var).values.numpy().copy())
        hist_dict['primal_var'].append(aug_lag.primal_var.cpu().numpy().copy())
        hist_dict['dual_var'].append(aug_lag.dual_var.cpu().numpy().copy())

        # print(f"Epoch {hist_dict['epoch'][-1]}:")
        # print(f"objective = {hist_dict['objective'][-1]}")
        # print(f"constraint = {hist_dict['constraint'][-1]}")

def pd_adam_quad_eq_solver(Q,A,b,theta_0,dual_0 = None, solver_params = None):

    params = {'lr_primal': 0.1, 'lr_dual':0.1, 'rho':10, 'steps': 100, 'pretrain_steps':0,'record_steps':0,'plots_folder':None}
    if solver_params is not None:
        params.update(solver_params)

    steps = params['steps']
    pretrain_steps = params['pretrain_steps']
    lr_primal = params['lr_primal']
    lr_dual = params['lr_dual']
    rho = params['rho']
    record_steps = params['record_steps']
    convergence_tol = params['convergence_tolerance']

    primal_var = torch.tensor(theta_0,requires_grad=True)
    primal_var.requires_grad = True

    Q = torch.tensor(Q)
    A = torch.tensor(A)
    b = torch.tensor(b)

    obj = Quad_Objective(Q)
    constr = LinEq_Constraint(A,b)

    if dual_0 is None:
        dual_var = torch.zeros(len(b))
    else:
        dual_var = torch.tensor(dual_0)

    dual_var.requires_grad = True

    aug_lag = Augmented_Lagrangian(obj,constr,primal_var,dual_var,rho)

    optimizer_primal = Adam([primal_var], lr=lr_primal,foreach=False)
    optimizer_dual = Adam([dual_var], lr=lr_dual, maximize = True,foreach=False)

    history_dict = {'epoch':[],
                    'objective': [],
                    'constraint': [],
                    'primal_var':[],
                    'dual_var':[],
                    'solver_params':params}
    
    record_history(history_dict,0,aug_lag)
    
    for step in range(1, steps + 1):
        pd_step(aug_lag, optimizer_primal, optimizer_dual)
        if record_steps != 0:
            if step % record_steps == 0 or step == steps:
                record_history(history_dict,step,
                        aug_lag)
                
        constraint = history_dict['constraint'][-1]
        grad_L = np.hstack((history_dict['primal_var'][-1]@Q.numpy() + history_dict['dual_var'][-1]*A.numpy(),constraint))

        if np.linalg.norm(grad_L,ord=np.inf) < convergence_tol:
            aug_lag.converged = True
            break

    return history_dict

# def pd_gd_quad_eq_solver(Q,A,b,x_0,lambda_0 = None, solver_params = None):

#     Q = Q.numpy()
#     A = A.numpy()
#     b = b.numpy()

#     params = {'lr_primal': 0.1, 'lr_dual':0.1, 'rho':10, 'steps': 100, 'pretrain_steps':0,'record_steps':0,'plots_folder':None}
#     if solver_params is not None:
#         params.update(solver_params)

#     steps = params['steps']
#     pretrain_steps = params['pretrain_steps']
#     lr_primal = params['lr_primal']
#     lr_dual = params['lr_dual']
#     rho = params['rho']
#     record_steps = params['record_steps']

#     primal_var = np.array(x_0)

#     if lambda_0 is None:
#         if pretrain_steps > 0:
#             # can train with just the penalty and then compute the dual variables
#             dual_var = np.zeros(len(b))
#         else:
#             dual_var = np.zeros(len(b))
#     else:
#         dual_var = np.array(lambda_0)

#     history_dict = {'epoch':[],
#                     'objective': [],
#                     'constraint': [],
#                     'primal_var':[],
#                     'dual_var':[]}
    
#     for step in range(1, steps + 1):
#         primal_var_old = primal_var.copy()
#         dual_var_old = dual_var.copy()
#         primal_var,dual_var = [primal_var_old-lr_primal*(primal_var_old@Q + A*dual_var_old),dual_var_old+lr_dual*(A@primal_var_old - b)]
#         if record_steps != 0:
#             if step % record_steps == 0 or step == steps:
#                 history_dict['epoch'].append(step)
#                 history_dict['objective'].append(1/2 * np.sum(primal_var * (Q@primal_var)))
#                 history_dict['constraint'].append(A@primal_var - b)
#                 history_dict['primal_var'].append(primal_var)
#                 history_dict['dual_var'].append(dual_var)

#                 print(f"Epoch {history_dict['epoch'][-1]}:")
#                 print(f"objective = {history_dict['objective'][-1]}")
#                 print(f"constraint = {history_dict['constraint'][-1]}")

#     return history_dict
    