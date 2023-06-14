import numpy as np
from matplotlib import pyplot as plt
from gradshafranovsolver.utils.contour_array import gen_contour_array
from gradshafranovsolver.utils.interpolate_gradient import interpolate_gradient
from shapely.geometry import LineString

def gen_adjacent_point(mug, xg, zg, zlevels, contour_array, cont_number, mu0, x0, gradx, gradmu, which_side):

    mu_eqn = np.linspace(mug[0,0], mug[0,-1], mug.shape[1])
    x_eqn = x0 + gradx/gradmu*(mu_eqn - mu0)

    fig,ax = plt.subplots()
    cs = ax.contour(mug, xg, zg, levels = zlevels)
    ax.scatter(mu0, x0)
    ax.quiver(mu0, x0, gradmu, gradx)
    ax.plot(mu_eqn, x_eqn, 'r', alpha = 0.7)
    ax.set(xlim=[0,10], ylim = [0,10])

    ls1 = LineString(contour_array[contour_array[:,2]==cont_number+which_side][:,:2])
    ls2 = LineString(np.c_[mu_eqn, x_eqn])
    points = ls1.intersection(ls2)
    mu_int, x_int = points.x, points.y
    ax.scatter(mu_int, x_int)

    plt.show()
