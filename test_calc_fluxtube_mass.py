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

# Test file to demonstrate calculation of the mass between flux surfaces. 

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
constant_density = ma/volume*np.ones(xgrid.shape)

# analytic dmdpsi for constant density from Brunet (2021) note
alpha_0 = x0*(XMAX+aratio)/R_star
psi_levels_lower = psi_levels[psi_levels<psi_star/psi_0/alpha_0]
psi_levels_upper = psi_levels[psi_levels>psi_star/psi_0/alpha_0]
chi_upper = (1.0 - psi_levels_upper*(psi_0/psi_star))**1.5
xi_upper = 8.0 + 12.0*psi_levels_upper*(psi_0/psi_star) + 15.0*(psi_levels_upper*psi_0/psi_star)**2.0
dchi_upper = -1.5*np.sqrt(1.0 - psi_levels_upper*psi_0/psi_star)
dxi_upper = 12.0+ 30.0*psi_levels_upper*(psi_0/psi_star)


analytic_dmdpsi_upper = -4.0*np.pi*constant_density[0,0]/105.0*R_star**3.0/psi_star*((psi_0/psi_star*psi_levels_upper)**3.*(dchi_upper*xi_upper + chi_upper*dxi_upper) - 3*(psi_levels_upper*psi_0/psi_star)**2*chi_upper*xi_upper)/((psi_levels_upper*psi_0/psi_star)**6)

chi_lower = (1.0 - psi_levels_lower*(psi_0/psi_star))**1.5
xi_lower = 8.0 + 12.0*psi_levels_lower*(psi_0/psi_star) + 15.0*(psi_levels_lower*psi_0/psi_star)**2.0
dchi_lower = -1.5*np.sqrt(1.0 - psi_levels_lower*psi_0/psi_star)
dxi_lower = 12.0+ 30.0*psi_levels_lower*(psi_0/psi_star)
alph_chi = (1.0 - alpha_0*psi_levels_lower*(psi_0/psi_star))**1.5
alph_xi = 8.0 + 12.0*alpha_0*psi_levels_lower*(psi_0/psi_star) + 15.0*(alpha_0*psi_levels_lower*psi_0/psi_star)**2.0
alph_dchi = -1.5*alpha_0*np.sqrt(1.0 - alpha_0*psi_levels_lower*psi_0/psi_star)
alph_dxi = 12.0*alpha_0+ 30.0*alpha_0**2.*psi_levels_lower*(psi_0/psi_star)

analytic_dmdpsi_lower = -4.*np.pi*constant_density[0,0]*R_star**3./105./psi_star*((psi_levels_lower*psi_0/psi_star)**3.*(dchi_lower*xi_lower + chi_lower*dxi_lower) - 3.*(psi_levels_lower*psi_0/psi_star)**2. * chi_lower * xi_lower - (psi_levels_lower*psi_0/psi_star)**3*(alph_dchi*alph_xi + alph_chi*alph_dxi) + 3.*alph_chi*alph_xi*(psi_levels_lower*psi_0/psi_star)**2.)/(psi_levels_lower*psi_0/psi_star)**6.

# calculate dmdpsi
dmdpsi = x0**3/psi_0*calc_dmdpsi(psi_grid, contour_array, constant_density, n_contours, aratio, alpha_ratio, dmu, dx, 1, 0)
# plot comparison
fig,ax = plt.subplots()
ax.plot(psi_levels_lower*psi_0/psi_star, analytic_dmdpsi_lower*psi_0/ma, label="Analytic lower")
ax.plot(psi_levels_upper*psi_0/psi_star, analytic_dmdpsi_upper*psi_0/ma, label="Analytic upper")
ax.plot(psi_levels*psi_0/psi_star, dmdpsi*psi_0/ma, '.', alpha = 0.4, label = "Numerical")
ax.axvline(R_star/x0/(XMAX + aratio), alpha = 0.4, linestyle='--', label="open-close boundary")
ax.set(xlabel=r"$\psi/\psi_*$", ylabel=r"$dM/d\psi$")

ax.legend()
plt.show()