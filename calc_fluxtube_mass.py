import numpy as np
from gradshafranovsolver.utils.gen_s import gen_s
from gradshafranovsolver.utils.interpolate_gradient import interpolate_gradient
from gradshafranovsolver.utils.interpolate_function import interpolate_function
from gradshafranovsolver.utils.integrand_function import integrand_function
from scipy import integrate


def calc_dmdpsi(psi_grid, contour_array, density, n_contours,aratio, alpha_ratio,dmu,dx, LOGSCALEX, X1LOW):

    sum_simps = np.zeros(n_contours)

    # loop over flux surfaces
    for index in range(n_contours):

        # select ith contour
        mucoords = contour_array[contour_array[:, 2] == index][:, 0]
        xcoords = contour_array[contour_array[:, 2] == index][:, 1]
        
        # calculate s
        s = gen_s(mucoords, xcoords, alpha_ratio, aratio)

        # interpolate grad(psi) and density along contour
        grad_psi_interp = np.zeros(len(mucoords))
        density_interp = np.zeros(len(mucoords))

        for i in range(len(grad_psi_interp)):
            gradx,gradmu = interpolate_gradient(psi_grid,mucoords[i],xcoords[i],dmu,dx,LOGSCALEX, X1LOW)
            grad_psi_interp[i] = np.sqrt(gradx*gradx + (1.0 - mucoords[i]*mucoords[i])/(xcoords[i] + alpha_ratio*aratio)/(xcoords[i] + alpha_ratio*aratio)*gradmu*gradmu)
            density_interp[i] = interpolate_function(density, mucoords[i],xcoords[i], dmu,dx, LOGSCALEX, X1LOW)

        # construct integrand
        integrand_vals = integrand_function(xcoords, mucoords, grad_psi_interp, aratio, 1, density_interp,alpha_ratio)

        # simpson's integration
        sum_simps[index] = integrate.simps(integrand_vals, s, even="first")

    return sum_simps