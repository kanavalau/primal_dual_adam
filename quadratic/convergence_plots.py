import numpy as np
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

pgf_with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 13,
    "font.size": 13,
    "legend.fontsize": 10,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 13,
    "pgf.rcfonts": False,
    "text.latex.preamble": 
        r'\usepackage{xcolor}',
    "pgf.preamble": 
        r'\usepackage{xcolor}'
}
matplotlib.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
MATHFONTSIZE = 14

import sys
sys.path.insert(0, '../utilities')
from quad_prob_def import quad_prob,compute_bounds

Q,A,b,_,_ = quad_prob()

rho = 2.0
alpha_bound,gamma_bound = compute_bounds(Q,A,rho)

conv_points_data = np.load('convergence_points.npz')
boundary_points_data = np.load('boundary_points.npz')

boundary_points = boundary_points_data['array1']
AS = conv_points_data['array1']
GS = conv_points_data['array2']
conv_points = conv_points_data['array3']

gammas_flat = GS.flatten()
alphas_flat = AS.flatten()
conv_flat = conv_points.flatten()

# Create plot
fig, ax = plt.subplots(figsize=(4,3))
# for i in range(len(alphas_flat)):
#     ax.scatter(alphas_flat[i], gammas_flat[i], color='green' if conv_flat[i] == 1 else 'red',marker='o', s=1)

alpha_min = np.min(alphas_flat)
alpha_max = np.max(alphas_flat)
gamma_min = np.min(gammas_flat)
gamma_max = np.max(gammas_flat)

ax.plot(boundary_points[0],boundary_points[1],':k')
x_shaded = boundary_points[0]
y_shaded = boundary_points[1]
ax.fill_between(x_shaded, gamma_min, y_shaded, where=(y_shaded >= gamma_min), color='grey', alpha=0.8)

ax.plot([alpha_bound,alpha_bound], [0,gamma_bound],'--k')
ax.plot([0,alpha_bound], [gamma_bound,gamma_bound],'--k')
ax.fill_between([alpha_bound,alpha_bound,0], 0, [0,gamma_bound,gamma_bound], where=(np.array([0,gamma_bound,gamma_bound]) >= 0), color='k', alpha=0.8)

ax.set_ylabel(r'$\gamma$')
ax.set_xlabel(r'$\alpha$')
ax.set_xscale('log')
ax.set_xlim((alpha_min,alpha_max))
ax.set_ylim((gamma_min,gamma_max))
ax.set_yscale('log')
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='s', color='w', label='Converges in simulation', markerfacecolor='grey', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Theoretical upper bound derived', markerfacecolor='k', markersize=10)]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=1)

plt.tight_layout()
plt.savefig('plots/convergence.pdf',bbox_inches='tight',pad_inches = 0)
plt.show()

plt.show()
