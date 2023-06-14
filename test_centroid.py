import numpy as np
from matplotlib import pyplot as plt

Lx = 10
Ly = 5

x0, y0 = 0,0

theta_angle = 78
theta = theta_angle*np.pi/180
phi = np.pi/2 - theta

x1 = x0 + Lx*np.cos(theta)
y1 = y0 + Lx*np.sin(theta)

x2 = x0 - Ly*np.cos(phi)
y2 = y0 + Ly*np.sin(phi)

x3 = x2 + Lx*np.cos(theta)
y3 = y2 + Lx*np.sin(theta)

xpoints = [x0, x1, x2, x3]
ypoints = [y0,y1,y2,y3]

avg_x = np.mean(xpoints)
avg_y = np.mean(ypoints)

fig,ax = plt.subplots()
ax.scatter(xpoints,ypoints)
ax.scatter(avg_x, avg_y, c= 'r')
ax.set(aspect='equal')
plt.show()

