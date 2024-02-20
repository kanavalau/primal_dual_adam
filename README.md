# Analysis and Local Convergence Proof of a Constrained Optimization Algorithm for Training Neural Networks

Andrei Kanavalau, Sanjay Lall

Stanford University

## Installation

To run the code, install the required packages using Conda:

```bash
conda env create -f environment.yml
conda activate primal_dual_adam
```

## Code overview
* quadratic/ contains the code for section 4.1
* aug_lag_plots.py produces figure 1
* trajectories_plots.py produces figure 2
* convergence_points.py sweeps through alpha and gamma
* convergence_binary_search.py uses binary search to find the convergence boundary
* these two scripts can take a while to run depending on the sampling density, the data generated for figure 3 is included in boundary_points.npz
* convergence_plots.py produces figure 3
* nn/ contains the code for section 4.2
* data_gp.py generates the training data
* fit_nns.py trains the three models
* nn_training_plots.py produces figure 4
* utilities/ contains the required functions including modified Adam