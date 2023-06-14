# tests
Contains various miscellaneous scripts used to test parts of the Grad-Shafranov solver code.

## Log of files:
These are the main test files worth describing. There are several other files which are minor tests in comparison, and so are neglected here:
- `calc_fluxtube_mass.py`: function that calculates mass-flux distribution $dM/d\psi$. Input to `test_calc_fluxtube_mass.py`
- `test_calc_fluxtube_mass.py`: script comparing analytic and numerical calculations of $dM/d\psi$
- `gen_adjacent_point.py`: finds point on adjacent contour normal to selected point
- `test_contour_vectors.py`: plots the gradient vector $\nabla f$ at a given contour of the scalar field $f(x,y)$ in Cartesian

The folders are:
- `magnetic_diffusion_solver`: simple code to solve the time-dependent magnetic diffusion equation in 2D
- `bayesian_modelling_in_python`: Jupyter notebook following the Bayesian Python tutorial [here](https://github.com/ryan-brunet/bayesian_modelling_in_python_ipynb)
- `alternative_dmdpsi_profiles_test`: testing the exponentially modified Gaussian as an alternative profile for $dM/d\psi$
