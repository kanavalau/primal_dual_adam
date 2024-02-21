import numpy as np
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

pgf_with_latex = {
    "text.usetex": True,            # use LaTeX to write all text
    "font.family": "serif",
    "axes.labelsize": 13,
    "font.size": 13,
    "legend.fontsize": 10,
    "axes.titlesize": 13,           # Title size when one figure
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 13,         # Overall figure title
    "pgf.rcfonts": False,
    "text.latex.preamble": 
        r'\usepackage{xcolor}',
    "pgf.preamble": 
        r'\usepackage{xcolor}'
}
matplotlib.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
MATHFONTSIZE = 14

def plot_aug_lag(Q,A,b,rho,theta_star,dual):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)

    X, Y = np.meshgrid(x, y)
    quad_term = np.einsum('ij,ij->j', np.dot(Q, np.vstack((X.ravel(), Y.ravel()))), np.vstack((X.ravel(), Y.ravel()))).reshape(X.shape)
    constraint = (np.dot(A, np.vstack((X.ravel(), Y.ravel()))) - b).reshape(X.shape)
    
    AL =1/2*quad_term + dual*constraint + rho/2*constraint**2

    plt.figure(figsize=(4, 3))

    contours = plt.contour(X, Y, AL, levels=7, colors='black', alpha=0.85)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

    plt.plot(theta_star[0],theta_star[1],'k*')

    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(f'plots/rho{rho:.0f}.pdf',bbox_inches='tight',pad_inches = 0)

def plot_history(history_dict):
    fig,axs = plt.subplots(2,sharex=True)

    axs[0].set_title('objective')
    axs[0].plot(history_dict['objective'],label='objective')

    constraint = np.vstack(history_dict['constraint'])
    idx = np.argmax(np.abs(constraint), axis=1)
    rows = np.arange(len(idx))
    constraint_max_vio = constraint[rows, idx]
    
    axs[1].set_title('constraint')
    axs[1].plot(constraint_max_vio)

    plt.show()

def plot_contours_and_feasibility(Q,A,b,history_dicts):

    x_vec = np.linspace(-0.5,2,1000)
    y_vec = np.linspace(0.5,2,1000)
    x, y = np.meshgrid(x_vec,y_vec)
    xy = np.vstack((x.flatten(),y.flatten()))

    _,ax = plt.subplots(figsize=(4,3.5))

    contours = ax.contour(x, y, 1/2*np.sum(xy*(Q@xy),axis=0).reshape(x.shape), levels=7, colors='black', alpha=0.85)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

    if A[1] == 0:
        ax.plot((b - A[1]*y_vec)/A[0],y_vec,'-.k',label='Primal feasibility')
    else:
        ax.plot(x_vec,(b - A[0]*x_vec)/A[1],'-.k',label='Primal feasibility')

    dual_feas_dir = np.linalg.inv(Q)@A.T
    if dual_feas_dir[0] == 0:
        ax.plot(y_vec*dual_feas_dir[0]/dual_feas_dir[1],y_vec,':k',label='Dual feasibility')
    else:
        ax.plot(x_vec,x_vec*dual_feas_dir[1]/dual_feas_dir[0],':k',label='Dual feasibility')

    colors = ['blue','green','purple','orange']
    for i,hist_dict in enumerate(history_dicts):
        primal_iterates = np.vstack(hist_dict['primal_var'])
        alpha = hist_dict['solver_params']['lr_primal']
        gamma = hist_dict['solver_params']['lr_dual']/alpha

        alpha_sci = f'{alpha:.1e}'.split('e')
        mantissa = alpha_sci[0]
        exponent = int(alpha_sci[1])
        label_string = r'$\alpha=' + mantissa + r'\times 10^{' + str(exponent) + r'},\ ' + r'\gamma=' + f'{gamma:.1f}$'
        ax.plot(primal_iterates[:,0],primal_iterates[:,1],color=colors[i],label=label_string)

    ax.plot(primal_iterates[0,0],primal_iterates[0,1],'ko',markerfacecolor='w')
    ax.plot(primal_iterates[-1,0],primal_iterates[-1,1],'k*')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2)
    plt.tight_layout()
    plt.savefig('plots/paths.pdf',bbox_inches='tight',pad_inches = 0)