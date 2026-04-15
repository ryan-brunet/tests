import numpy as np
from numba import njit
import sys

# SOR algorithm with Chebyshev acceleration
# @njit
def sor(a,b,c,d,e,f,psi_grid,MAXITS,imax,jmax,w,tol,rjac):
    # anormf=0.0
    # anormf=np.sum(np.abs(f))
    w = 1.0

    for n in range(0,MAXITS):
            # print(f"iter = {n} psi[1,10] = {psi_grid[1,10]}")
            anorm = 0.0
            isw = 1
 
            for ipass in range(1,3):

                jsw=isw

                # Relaxation iteration procedure to update psi(x,y):
                for i in range(0,imax):
                    
                    for j in range(jsw-1,jmax,2):

                        residual = 0.0

                        # Calculate residuals as per SOR algorithm equation 19.5.28 Press et al 1992 and Notebook 04 p94
                        if(i<imax-1):
                            residual+=a[i,j]*psi_grid[i+1,j]

                        if(i>0):
                            residual+=b[i,j]*psi_grid[i-1,j]

                        if(j<jmax-1):
                            residual+=c[i,j]*psi_grid[i,j+1]

                        if(j>0):
                            residual+=d[i,j]*psi_grid[i,j-1]

                        residual+=e[i,j]*psi_grid[i,j]
                        
                        residual-=f[i,j]

                        anorm+=np.abs(residual)

#                         if((n < 20)):
#                             if(i < 2):
#                                 if((j < jmax-1)):
#                                     print(f"n={n} ({i}, {j}) psi={psi_grid[i,j]:0.6f} w*resid/e_ij = {w*residual/e[i,j]:0.6f} a ={a[i,j]*psi_grid[i+1,j]:0.6f} b={b[i,j]:0.6f} c={c[i,j]*psi_grid[i,j+1]:0.6f}, d={d[i,j]:0.6f} e={e[i,j]*psi_grid[i,j]:0.6f} f={f[i,j]}, r = {residual:0.6f}")
#                                     # print(f"n={n} ({i}, {j}) psi={psi_grid[i,j]} w*resid/e_ij = {w*residual/e[i,j]} resid={residual}, {a[i,j]*psi_grid[i+1,j] +c[i,j]*psi_grid[i,j+1] + e[i,j]*psi_grid[i,j] - f[i,j]}")
#                             else:
#                                 continue
                        

                        psi_grid[i,j] -= w*residual/e[i,j]

                    jsw = 3-jsw

                isw = 3 - isw
                
                if((n==1) & (ipass==1)):
                    w = 1.0/(1.0-0.5*rjac*rjac)
                else:
                    w = 1.0/(1.0-0.25*rjac*rjac*w)

            # print("iter = %d anorm = %lg"%(n,anorm/anormf))
                
            if(anorm < tol):
                # print("Finished")
                break
    return psi_grid