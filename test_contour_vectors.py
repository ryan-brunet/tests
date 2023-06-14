import sys
import numpy as np
from matplotlib import pyplot as plt
from gradshafranovsolver.utils.contour_array import gen_contour_array
from gradshafranovsolver.utils.interpolate_gradient import interpolate_gradient
from shapely.geometry import LineString

point_number = 60
cont_number = 1
which_side = -1

xmin = 0.1
xmax = 10
mumin = 0.0
mumax = 1.0
nx = 101
nmu = 101

LOGSCALEX = 1
X1LOW = 0
X1 = np.log(np.exp(X1LOW)*xmax + 1.0)
DX1 = X1/(nx - 1.0)

if(LOGSCALEX == 1):
    ii = np.linspace(0,nx-1,nx)
    x = np.exp(-X1LOW)*(np.exp(ii*DX1)-1)
else:
    x = np.linspace(xmin,xmax,nx)
        
mu = np.linspace(mumin,mumax,nmu)

mug, xg = np.meshgrid(mu, x)
dx = x[1] - x[0]
dmu = mu[1] - mu[0]

zg = (1-mug**2)/(xg + 1)#np.sqrt(xg**2 + mug**2)
zlevels = np.linspace(np.min(zg) + 0.01, np.max(zg) - 0.01, 14)

# get contour coordinates
contour_array = gen_contour_array(xg, mug, zg, zlevels)

muc = contour_array[contour_array[:,2]==cont_number][:,0]
xc = contour_array[contour_array[:,2]==cont_number][:,1]

gradx, gradmu = interpolate_gradient(zg, muc[point_number],xc[point_number],dmu,dx, LOGSCALEX,0)

# construct line passing through point
mu0 = muc[point_number]
x0 = xc[point_number]

xxg = xg*np.sqrt(1.0-mug**2)
yyg = xg*mug
xx0 = x0*np.sqrt(1.0 - mu0**2)
yy0 = x0*mu0

cartesian_contour_array = gen_contour_array(yyg, xxg, zg, zlevels)

xc2 = cartesian_contour_array[cartesian_contour_array[:,2]==cont_number][:,0]
yc2 = cartesian_contour_array[cartesian_contour_array[:,2]==cont_number][:,1]

xc3 = xc*np.sqrt(1.0-muc**2)
yc3 = xc*muc

gradxx = np.sqrt(1.0 - mu0**2)*gradx - mu0*np.sqrt(1-mu0**2)/(x0+1)*gradmu
gradyy = mu0*gradx + (1-mu0**2)/(x0+1)*gradmu
x_eqn = np.linspace(0, 10, 101)
y_eqn = yy0 + gradyy/gradxx*(x_eqn - xx0)

fig,ax = plt.subplots()
ax.contour(xxg,yyg,zg, levels = zlevels)
ax.quiver(xx0,yy0, gradxx, gradyy)
plt.gca().set_aspect('equal')
ax.scatter(xx0,yy0)
ax.scatter(xc3,yc3)
ax.plot(x_eqn, y_eqn, 'r', alpha = 0.7)

ls1 = LineString(cartesian_contour_array[cartesian_contour_array[:,2]==cont_number+which_side][:,:2])
ls2 = LineString(np.c_[x_eqn, y_eqn])
points = ls1.intersection(ls2)
x_int, y_int = points.x, points.y
ax.scatter(x_int, y_int)
ax.set(xlim=[0,10], ylim = [0,10])

# now convert back to x, mu coordinates
xc_int = np.sqrt(x_int**2 + y_int**2)
muc_int = np.cos(np.arctan(x_int/y_int))

fig,ax = plt.subplots()
cs = ax.contour(mug, xg, zg, levels = zlevels)
ax.scatter(mu0, x0)
ax.quiver(mu0, x0, gradmu, gradx)
ax.scatter(muc_int, xc_int)
plt.gca().set_aspect('equal')

# ax.quiver(mu0,x0,-2*mu0/(x0+1), -(1-mu0**2)/(x0+1)**2)
# cs = ax.contour(np.arccos(mug), np.log10(xg), zg, levels = zlevels)
# ax.scatter(np.arccos(mu0), np.log10(x0))
# ax.quiver(np.arccos(mu0), np.log10(x0), -np.sqrt(1-mu0**2)/(x0)*gradmu, gradx)
# ax.plot(mu_eqn, x_eqn, 'r', alpha = 0.7)
# ax.set(xlim=[0,1], ylim = [0,1])

plt.show()

