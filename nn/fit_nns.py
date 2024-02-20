import sys
sys.path.insert(0, '../utilities')

import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
from adam import Adam
from aug_lag_def import Augmented_Lagrangian_NN,MSE_Objective, No_noise_points_constraint
from pd_adam_solver import pd_step
import numpy as np

torch.manual_seed(1)

def record_history(hist_dict,step,aug_lag):
    # Appends the data from the most recent iteration to a dictionary
    with torch.no_grad():
        hist_dict['epoch'].append(step)
        hist_dict['objective'].append(aug_lag.objective().item())
        if torch.is_tensor(aug_lag.constraint()):
            hist_dict['constraint'].append(aug_lag.constraint().numpy().copy())
        else:
            hist_dict['constraint'].append(aug_lag.constraint().values.numpy().copy())
        hist_dict['dual_var'].append(aug_lag.dual_var.cpu().numpy().copy())

        print(f"Epoch {hist_dict['epoch'][-1]}:")
        print(f"objective = {hist_dict['objective'][-1]}")
        print(f"constraint = {hist_dict['constraint'][-1]}")

class func_fit(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 6)
        self.layer2 = nn.Linear(6, 6)
        self.output_layer = nn.Linear(6, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.output_layer(x)
    
def fit_nn(x_train,u_train,x_no_noise,u_no_noise,alpha,gamma,rho):
    model = func_fit()

    obj = MSE_Objective(model,x_train,u_train)
    constr = No_noise_points_constraint(model,x_no_noise,u_no_noise)

    dual_var = torch.zeros(len(constr()))
    dual_var.requires_grad = True
    aug_lag = Augmented_Lagrangian_NN(model, obj, constr, dual_var, rho)

    optimizer_objective = Adam(model.parameters(), lr=alpha)
    optimizer_dual = Adam([aug_lag.dual_var], lr=alpha*gamma, maximize = True)

    history_dict = {'epoch':[],
                    'objective': [],
                    'constraint': [],
                    'dual_var':[]}

    for epoch in range(1, epochs + 1):
        pd_step(aug_lag, optimizer_objective,optimizer_dual)
        if epoch % hist_out == 0 or epoch == epochs:
            record_history(history_dict,epoch,
                        aug_lag)
        aug_lag.constraint()

    return history_dict

data = np.load('data_gp.npz')
x = data['array1']
u = data['array2']

no_noise_subsample_rate = 4
epochs = 550000
hist_out = 5000
alpha = 2*10**(-3)

x_train = torch.tensor(x).reshape((-1,1))
u_train = torch.tensor(u).reshape((-1,1))

x_no_noise = x_train[::no_noise_subsample_rate]
u_no_noise = u_train[::no_noise_subsample_rate]

u_train += 0.1*torch.randn(u_train.shape)
u_train[::no_noise_subsample_rate] = u_no_noise

# Unconstrained
gamma = 0
rho = 0
history_dict_unc = fit_nn(x_train,u_train,x_no_noise,u_no_noise,
                                    alpha,gamma,rho)

# Penalty
gamma = 0
rho = 10
lr_dual = alpha*gamma
history_dict_pen = fit_nn(x_train,u_train,x_no_noise,u_no_noise,
                                    alpha,gamma,rho)

# Constr
rho = 10
gamma = 10**(-4)
lr_dual = alpha*gamma
history_dict_con = fit_nn(x_train,u_train,x_no_noise,u_no_noise,
                                    alpha,gamma,rho)

import pickle

with open('training_history.pkl', 'wb') as file:
    pickle.dump((history_dict_unc, history_dict_pen, history_dict_con), file)
