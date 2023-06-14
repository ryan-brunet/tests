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

# Test file to demonstrate calculation of the mass between flux surfaces

# initialise constants
config_file = "/Users/ryanbrunet/Documents/phd/code/gradshafranovsolver/gs-config.yaml"

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
# initialise dipole flux function
psi_grid, psi_levels = psi_gen(xgrid,mugrid,psi_star,psi_0,n_contours,aratio,dx,dmu,alpha_ratio)

# generate contour array
contour_array = gen_contour_array(xgrid,mugrid,psi_grid,psi_levels)
# initialise density grid
volume = 0.5*(4.0/3.0*np.pi*x0**3*(XMAX + alpha_ratio)**3 - aratio**3)
density = ma/volume*np.ones(xgrid.shape)

# analytic ratio from Brunet (2021) note
chi = (1.0 - psi_levels*(psi_0/psi_star))**1.5
xi = 8.0 + 12.0*psi_levels*(psi_0/psi_star) + 15.0*(psi_levels*psi_0/psi_star)**2.0
dchi = -1.5*np.sqrt(1.0 - psi_levels*psi_0/psi_star)
dxi = 12.0+ 30.0*psi_levels*(psi_0/psi_star)
analytic_dmdpsi = -4.0*np.pi*density[0,0]/105.0*R_star**3.0/psi_star/(psi_0/psi_star)**4*((psi_0/psi_star)*psi_levels**3*(dchi*xi + chi*dxi) - 3*psi_levels**2*chi*xi)/(psi_levels**6)

# calculate dmdpsi
dmdpsi = x0**3/psi_0*calc_dmdpsi(psi_grid, contour_array, density, n_contours, aratio, alpha_ratio, dmu, dx, 1, 0)
# plot comparison
fig,ax = plt.subplots()
ax.plot(np.log10(psi_levels), np.log10(analytic_dmdpsi), label="Analytic")
ax.plot(np.log10(psi_levels), np.log10(dmdpsi), '.', alpha = 0.4, label = "Numerical")
ax.set(xlabel=r"$\tilde{\psi} = \psi/\psi_0$", ylabel=r"$dM/d\psi$")
ax.legend()
plt.show()