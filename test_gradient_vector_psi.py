import sys
import numpy as np
import ruamel.yaml as yaml
from gradshafranovsolver.utils.grid_gen import grid_gen
from gradshafranovsolver.utils.psi_gen import psi_gen
from gradshafranovsolver.utils.contour_array import gen_contour_array
from calc_fluxtube_mass import calc_dmdpsi
from matplotlib import pyplot as plt
plt.rc('text',usetex=True)
plt.rc('font', family='serif')
from scipy.integrate import simpson
from gradshafranovsolver.utils.gen_s import gen_s
from gradshafranovsolver.gen_adjacent_point import gen_adjacent_point
from gradshafranovsolver.utils.interpolate_function import interpolate_function
from gradshafranovsolver.utils.interpolate_gradient import interpolate_gradient

# Test file to plot normal vector in order to calculate intersection with adjacent contour. Rather than converting to cartesian, we can just do everything in (x,mu) coordinates. Note that, whilst visually the curves do not look like they project normal to the flux lines, this is a result of the scaling of the plotting window and the fact we are doing things in log(x) etc scaling. Don't worry, follow the maths.

# initialise constants
config_file = "gs-config.yaml"

with open(config_file) as file:
    dictionary = yaml.load(file, Loader=yaml.Loader)

nr = dictionary['nr']
nth = dictionary['nth']
XMIN = dictionary['XMIN']
XMAX = dictionary['XMAX']
MUMIN = dictionary['MUMIN']
MUMAX = dictionary['MUMAX']
X1LOW = dictionary['X1LOW']
LOGSCALEX = dictionary['LOGSCALEX']
B_star = dictionary['B_star']
R_star = dictionary['R_star']
cs = dictionary['cs']
M_NS = dictionary['M_NS']
M_sol = dictionary['M_sol']
M_star = M_NS*M_sol
m_accretion = dictionary['m_accretion']
G = dictionary['G']
alpha_ratio = dictionary['alpha_ratio']
bratio=dictionary['bratio']

# calculate constants
x0 = cs*cs*R_star*R_star/(G*M_star)
aratio = R_star/x0
n_contours = int(nr-1)
ma = m_accretion*M_sol
x0 = cs*cs*R_star*R_star/(G*M_star)
X1 = np.log(np.exp(X1LOW)*XMAX + 1.0)
DX1 = X1/(nr - 1.0)
psi_star = B_star*R_star**2/2
psi_0 = np.sqrt(cs*cs*ma/x0/x0/x0)

if(LOGSCALEX == 1):
    dx = DX1
else:
    dx = (XMAX-XMIN)/(nr-1.0)
dmu = np.abs(MUMAX-MUMIN)/(nth-1.0)

# initialise grid
mugrid,xgrid = grid_gen(XMIN,XMAX,MUMIN,MUMAX,nr,nth,LOGSCALEX,DX1,X1LOW)
theta = np.arccos(mugrid[1:,:])
xlog = np.log10(xgrid[1:,:])

# initialise dipole flux function
psi_grid, psi_levels = psi_gen(xgrid,mugrid,psi_star,psi_0,n_contours,aratio,dx,dmu,alpha_ratio)

# Import data:
#============================================================================================================#
# Paths to files to read

selected_path = "reference_run"
selected_file = "reference_run"

# Path to desired output files

selected_readpath = "../gradshafranovsolver/output_files/" + selected_path + "/"

# Psi
filein_selected = selected_readpath + "Psi_output_" + selected_file + ".txt"
print(f"Results file: {filein_selected}")
data_selected = np.loadtxt(filein_selected)
psi_data_selected = data_selected[:,2]
N = int(np.sqrt(len(psi_data_selected)))
psi_selected = np.reshape(psi_data_selected,(N,N), order = 'C')

# Psi levels
file = selected_readpath + "psi_levels_output.txt"
psi_levels_file = np.loadtxt(file)

# generate contour array
contour_array = gen_contour_array(xgrid,mugrid,psi_selected,psi_levels)

for index in range(3,n_contours-2):
        
    # Generate (r, theta) coordinates for the selected flux surface
    xcoords = contour_array[contour_array[:, 2] == index][:, 1]
    mucoords = contour_array[contour_array[:, 2] == index][:, 0]

    # Calculate arc-length along flux surface:
    # Generate s-coordinate array:
    s = gen_s(mucoords, xcoords,alpha_ratio, aratio)

    gradx = np.zeros_like(s)
    gradmu = np.zeros_like(s)

    for k in range(len(s)):
        gradx[k], gradmu[k] = interpolate_gradient(psi_selected,mucoords[k],xcoords[k],dmu,dx,LOGSCALEX,X1LOW)

    k = int(0.8*len(s))
    vector_mu_component = -np.sqrt(1 - mucoords[k]**2)/(xcoords[k] + aratio)*gradmu[k]
    vector_x_component = -gradx[k]
    vector_mu_vals = np.linspace(0, 1, 100)
    vector_gradient = vector_x_component / vector_mu_component
    vector_proj = vector_gradient*vector_mu_vals + (xcoords[k] - vector_gradient*mucoords[k])

    # plot
    fig,ax = plt.subplots()
    ax.contour(mugrid,xgrid, psi_selected, levels=psi_levels)
    ax.scatter(mucoords[k], xcoords[k])
    ax.quiver(mucoords[k], xcoords[k], vector_mu_component, vector_x_component)
    ax.plot(vector_mu_vals, vector_proj, 'r',alpha = 0.5)
    plt.show()
