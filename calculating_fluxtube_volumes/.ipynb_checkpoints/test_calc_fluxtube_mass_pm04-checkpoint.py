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

# Test file to calculate the amount of mass between flux surfaces for the PM04 mass-flux distribution. 

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
theta = 180./np.pi*np.arccos(mugrid[1:,:])
xlog = np.log10(xgrid[1:,:])

# initialise dipole flux function
psi_grid, psi_levels = psi_gen(xgrid,mugrid,psi_star,psi_0,n_contours,aratio,dx,dmu,alpha_ratio)

# Import data:
#============================================================================================================#
# Paths to files to read

selected_path = "test_ma_1e-8"
selected_file = "test_ma_1e-8"

# Path to desired output files

selected_readpath = "../gradshafranovsolver/output_files/" + selected_path + "/"

# Psi
filein_selected = selected_readpath + "Psi_output_"+ selected_file + ".txt"
print(f"Results file: {filein_selected}")
data_selected = np.loadtxt(filein_selected)
psi_data_selected = data_selected[:,2]
N = int(np.sqrt(len(psi_data_selected)))
psi_selected = np.reshape(psi_data_selected,(N,N), order = 'C')
# Psi levels
file = selected_readpath + "psi_levels_output.txt"
psi_levels = np.loadtxt(file)

# generate contour array
contour_array = gen_contour_array(xgrid,mugrid,psi_selected,psi_levels)

# Density
filein9= selected_readpath + "density_output_" + selected_file +".txt"
data9 = np.loadtxt(filein9)
densitydata = data9[:,2]
density = np.reshape(densitydata,(N,N), order='C')
# calculate dmdpsi for PM04
dmdpsi = x0**3/psi_0*calc_dmdpsi(psi_selected, contour_array, density, n_contours, aratio, alpha_ratio, dmu, dx, 1, 0)

# analytic form of dmdpsi PM04
dmdpsi_pm04 = 0.5*psi_0/psi_star*bratio*np.exp(-bratio*psi_levels*psi_0/psi_star)/(1.0-np.exp(-bratio))

# now check mass calculation using method in instability_calc script:
fluxtube_mass_a = np.zeros_like(psi_levels)
fluxtube_mass_b = np.zeros_like(psi_levels)
overall_fluxmass = np.zeros_like(psi_levels)

# Calculate mass in each fluxtube
for index in range(1,n_contours-2):
        
    # Generate (x, mu) coordinates for the selected flux surface
    xcoords = contour_array[contour_array[:, 2] == index][:, 1]
    mucoords = contour_array[contour_array[:, 2] == index][:, 0]

    # Calculate arc-length along flux surface:
    # Generate s-coordinate array:
    s = gen_s(mucoords, xcoords,alpha_ratio, aratio)

    # Preallocate arrays
    density_interp = np.zeros_like(s)
    density_interp_down = np.zeros_like(s)
    density_interp_next = np.zeros_like(s)
    density_interp_down_next = np.zeros_like(s)
    mass_a = np.zeros_like(s)
    mass_b = np.zeros_like(s)
    output_a = np.zeros_like(s)
    output_b = np.zeros_like(s)  
    overall_gradx = np.zeros_like(s)
    overall_gradmu = np.zeros_like(s)
    overall_density_interp = np.zeros_like(s)
    overall_density_interp_down = np.zeros_like(s)
    mucoords_down = np.zeros_like(s)
    xcoords_down = np.zeros_like(s)
    gradx_down = np.zeros_like(s)
    gradmu_down = np.zeros_like(s)

    for k in range(len(s)):
        # Find grad(psi) components along field line
        overall_gradx[k], overall_gradmu[k] = interpolate_gradient(psi_selected,mucoords[k],xcoords[k],dmu,dx,LOGSCALEX,X1LOW)

        # Generate (x,mu) coordinates on adjacent surface normal to (x,mu) 
        mucoords_down[k], xcoords_down[k], _ = gen_adjacent_point(contour_array, index, k, overall_gradx[k], overall_gradmu[k], -1, aratio, XMIN, XMAX, MUMIN, MUMAX, xgrid, mugrid, psi_selected, psi_levels, "point", "off")
        if((mucoords_down[k]<0)|(xcoords_down[k]<0)):
            continue

        # calculate gradients at each point (needed for calculation of volume) on the adjacent surface (already have grad_i psi (i = {mu, x}) on selected surface)
        gradx_down[k], gradmu_down[k] = interpolate_gradient(psi_selected,mucoords_down[k],xcoords_down[k],dmu,dx,LOGSCALEX, X1LOW)
        
        # interpolate density on both surfaces
        overall_density_interp[k] = interpolate_function(density, mucoords[k],xcoords[k], dmu,dx, LOGSCALEX, X1LOW)
        overall_density_interp_down[k] = interpolate_function(density, mucoords_down[k],xcoords_down[k], dmu,dx, LOGSCALEX, X1LOW)

    if(index%10==0):
        for i in range(len(s)):
            if(i%10==0):
                fig,ax3 = plt.subplots()
                ax3.contour(theta, xlog, psi_selected[1:,:], levels=psi_levels, alpha = 0.4)
                ax3.scatter(180./np.pi*np.arccos(mucoords), np.log10(xcoords), s= 2, alpha = 0.4,label="x,mu")
                ax3.scatter(180./np.pi*np.arccos(mucoords_down), np.log10(xcoords_down), s= 2, alpha = 0.4,label="x,mu -1")
                ax3.scatter(180./np.pi*np.arccos(mucoords_down[i]), np.log10(xcoords_down[i]),label="A")
                ax3.scatter(180./np.pi*np.arccos(mucoords[i]), np.log10(xcoords[i]), label="B")
                ax3.legend()
                ax3.set(xlim=[0,np.max(180./np.pi*np.arccos(mucoords))])
                plt.show()
    
    # combine gradient components to get grad(psi) on each surface
    overall_grad_psi_interp = np.sqrt(overall_gradx**2 + (1.0 - mucoords*mucoords)/(xcoords + alpha_ratio*aratio)/(xcoords + alpha_ratio*aratio)*overall_gradmu**2)
    overall_grad_psi_interp_down = np.sqrt(gradx_down**2 + (1.0 - mucoords_down*mucoords_down)/(xcoords_down + alpha_ratio*aratio)/(xcoords_down + alpha_ratio*aratio)*gradmu_down**2)
    
    # form overall integrand on each surface
    overall_integrand = (xcoords + aratio)*np.sqrt(1. - mucoords**2)*overall_density_interp/overall_grad_psi_interp
    overall_integrand_down = (xcoords_down + aratio)*np.sqrt(1. - mucoords_down**2)*overall_density_interp_down/overall_grad_psi_interp_down
    
    # calculate mass in each volume element for each fluxtube
    for j in range(len(s)-1):

        mass_a[j] = 2.*np.pi*x0**3*(psi_levels[index] - psi_levels[index-1])*simpson(overall_integrand_down[j:j+2], s[j:j+2])
        mass_b[j] = 2.*np.pi*x0**3*(psi_levels[index] - psi_levels[index-1])*simpson(overall_integrand[j:j+2], s[j:j+2])
    
    # sum up each fluxtube mass
    fluxtube_mass_a[index-1] = np.sum(mass_a)
    fluxtube_mass_b[index-1] = np.sum(mass_b)

    # compare to overall fluxtube mass
    overall_fluxmass[index] = 2*np.pi*x0**3*(psi_levels[index+1] - psi_levels[index])*simpson(overall_integrand, s, even="first")

    print(f"{index+1}/{n_contours}")

# plotting
fig,ax =plt.subplots()
ax.plot(psi_levels[:-1], fluxtube_mass_a[:-1]/((psi_levels[1:] - psi_levels[:-1])*ma),'.', label="mass A")
ax.plot(psi_levels[:-1], fluxtube_mass_b[:-1]/((psi_levels[1:] - psi_levels[:-1])*ma),'.', label="mass B")
ax.plot(psi_levels[:-1], overall_fluxmass[:-1]/((psi_levels[1:] - psi_levels[:-1])*ma), '.', label="overall")
ax.plot(psi_levels[:-1], dmdpsi_pm04[:-1],alpha=0.4, label="PM04")
plt.legend()
plt.show()

