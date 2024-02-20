import pickle

with open('training_history.pkl', 'rb') as file:
    history_dict_unc, history_dict_pen, history_dict_con = pickle.load(file)

import numpy as np
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

pgf_with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 12,
    "pgf.rcfonts": False,
    "text.latex.preamble": 
        r'\usepackage{xcolor}',
    "pgf.preamble": 
        r'\usepackage{xcolor}'
}
matplotlib.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
MATHFONTSIZE = 14

fig,axs = plt.subplots(2,sharex=True,figsize=(4,4))

axs[0].set_ylabel('MSE')
axs[0].set_yscale('log')
axs[0].plot(history_dict_unc['epoch'],history_dict_unc['objective'],color='xkcd:black',label='Unconstrained')
axs[0].plot(history_dict_pen['epoch'],history_dict_pen['objective'],color='xkcd:orange',label='Penalty')
axs[0].plot(history_dict_con['epoch'],history_dict_con['objective'],color='xkcd:purple',label='Constrained')
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)
constraint_nom = np.vstack(history_dict_unc['constraint'])
constraint_pen = np.vstack(history_dict_pen['constraint'])
constraint_constr = np.vstack(history_dict_con['constraint'])

axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Constraint')
axs[1].set_yscale('log')
axs[1].plot(history_dict_unc['epoch'],np.max(np.abs(constraint_nom), axis=1),color='xkcd:black',label='Unconstrained')
axs[1].plot(history_dict_pen['epoch'],np.max(np.abs(constraint_pen), axis=1),color='xkcd:orange',label='Penalty')
axs[1].plot(history_dict_con['epoch'],np.max(np.abs(constraint_constr), axis=1),color='xkcd:purple',label='Constrained')
plt.tight_layout()
plt.savefig('plots/training.pdf',bbox_inches='tight',pad_inches = 0)
print('Constraint violation unconstrained')
print(np.max(np.abs(constraint_nom), axis=1)[-1])
print('Constraint violation penalty')
print(np.max(np.abs(constraint_pen), axis=1)[-1])
print('Constraint violation constrained')
print(np.max(np.abs(constraint_constr), axis=1)[-1])