from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib       import rc, rcParams
import noisyduck as nd
#import noisyduck.annulus


# Define mean state
gam = 1.4
rho = 1.2212179
u   = 0.
v   = 0.
w   = 103.2586448
p   = 103341.6659

# Geometry
ri    = 0.25
ro    = 1.0

# Define circumferential and temporal wavenumber
omega = 3441.9548
m     = 2


# Numerical decomposition
eigenvalues_n, eigenvectors_n, r_n = nd.annulus.numerical.decomposition(omega,m,ri,ro,rho,u,v,w,p,gam,filter='acoustic')


# Nondimensionalize for analytical case
c = np.sqrt(gam*p/rho)
mach = w/c
omega = omega/c
sigma = ri
n = 20

# Analytical decomposition
eigenvalues_a, eigenvectors_a, r_a = nd.annulus.analytical.decomposition(omega,m,mach,ri,ro,n)





# Font/plot setup
rc('text', usetex=True)
rc('font', family='serif')
rcParams.update({'figure.autolayout': True})
fig1 = plt.figure()
fig1.set_facecolor('White')
ax1 = fig1.add_subplot(111)

# Plot numerical eigenvalues
for i in range(len(eigenvalues_n)):
    if (eigenvalues_n[i].imag > 0.):
        h_up, = ax1.plot(eigenvalues_n[i].real,eigenvalues_n[i].imag,   'b^',markersize=5, label='Acc. Down')
    elif (eigenvalues_n[i].imag < 0.):
        h_down, = ax1.plot(eigenvalues_n[i].real,eigenvalues_n[i].imag, 'bs',markersize=5, label='Acc. Up'  )

# Plot analytical eigenvalues
h_analytical, = ax1.plot(eigenvalues_a.real,eigenvalues_a.imag, 'ko', markerfacecolor='None',markersize=10,label='Analytical')


# Axis settings
ax1.legend(handles=[h_analytical,h_up,h_down],numpoints=1)
ax1.set_xlabel('$Re(k_z)$')
ax1.set_ylabel('$Im(k_z)$')

plt.show()


