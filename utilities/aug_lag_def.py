import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

class Augmented_Lagrangian(nn.Module):
    # Construct and augmented lagrangian for given objective, constraints, lagrange multipliers, and penalty constants

    def __init__(self,objective,constraint,primal_var,dual_var,rho):
        super(Augmented_Lagrangian, self).__init__()
        self.objective = objective
        self.constraint = constraint
        self.rho = rho
        self.primal_var = primal_var
        self.dual_var = dual_var
        self.converged = False

    def forward(self):
        obj = self.objective(self.primal_var)
        constr = self.constraint(self.primal_var)
        return obj + torch.dot(self.dual_var,constr) + self.rho/2*torch.norm(constr,p = 2)**2
    
class Augmented_Lagrangian_NN(nn.Module):

    def __init__(self,model,objective,constraint,dual_var,rho):
        super().__init__()
        self.objective = objective
        self.constraint = constraint
        self.model = model
        self.rho = rho
        self.dual_var = dual_var
        self.converged = False

    def forward(self):
        obj = self.objective()
        constr = self.constraint()
        return obj + torch.dot(self.dual_var,constr) + self.rho/2*torch.norm(constr,p = 2)**2

class Quad_Objective(nn.Module):
    def __init__(self,Q):
        super().__init__()
        self.Q = Q

    def forward(self,x):
        return 1/2 * x @ self.Q @ x
    
class MSE_Objective(nn.Module):
    def __init__(self,model,x,u_target):
        super(MSE_Objective, self).__init__()
        self.model = model
        self.x = x
        self.u_target = u_target

    def forward(self):
        return torch.mean(torch.norm(self.model(self.x) - self.u_target,p=2,dim=1) ** 2)
    
class No_noise_points_constraint(nn.Module):
    
    def __init__(self,model,x_vals,u_vals):
        super().__init__()
        self.model = model
        self.x_vals = x_vals
        self.u_vals = u_vals

    def forward(self):
        u = self.model(self.x_vals)
        return (u - self.u_vals).flatten()

class LinEq_Constraint(nn.Module):

    def __init__(self,A,b):
        super().__init__()
        self.A = A
        self.b = b
        self.satisfied = False

    def forward(self,x):
        constr_val = self.eq_constraint(x)
        self.satisfied = (constr_val == 0).all()
        return constr_val
    
    def eq_constraint(self,x):
        return self.A @ x - self.b