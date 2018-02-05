from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import noisyduck as nd


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
r = np.linspace(ri,ro,50)

# Define circumferential and temporal wavenumber
omega = 3441.9548
m     = 2


# Numerical decomposition
eigenvalues_r, eigenvectors_r = nd.annulus.numerical.decomposition(omega,m,r,rho,u,v,w,p,gam,filter='None')
eigenvalues_n, eigenvectors_n = nd.annulus.numerical.decomposition(omega,m,r,rho,u,v,w,p,gam,filter='acoustic')

# Separate eigenvectors into primitive variables
res = len(r)
rho_eigenvectors = eigenvectors_n[0*res:1*res,:]
u_eigenvectors   = eigenvectors_n[1*res:2*res,:]
v_eigenvectors   = eigenvectors_n[2*res:3*res,:]
w_eigenvectors   = eigenvectors_n[3*res:4*res,:]
p_eigenvectors   = eigenvectors_n[4*res:5*res,:]





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
fig2 = plt.figure(figsize=(4.5,6))
fig2.set_facecolor('White')
ax2 = fig2.add_subplot(111)
fig3 = plt.figure(figsize=(4.5,6))
fig3.set_facecolor('White')
ax3 = fig3.add_subplot(111)


# Plot raw numerical eigenvalues
for i in range(len(eigenvalues_r)):
    if (eigenvalues_r[i].imag > 0.):
        ax1.plot(eigenvalues_r[i].real,eigenvalues_r[i].imag, 'b^',markersize=5)
    elif (eigenvalues_r[i].imag < 0.):
        ax1.plot(eigenvalues_r[i].real,eigenvalues_r[i].imag, 'bs',markersize=5)


# Plot numerical eigenvalues
for i in range(len(eigenvalues_n)):
    if (eigenvalues_n[i].imag > 0.):
        h_up, = ax2.plot(eigenvalues_n[i].real,eigenvalues_n[i].imag,   'b^',markersize=5, label='Acc. Down')
    elif (eigenvalues_n[i].imag < 0.):
        h_down, = ax2.plot(eigenvalues_n[i].real,eigenvalues_n[i].imag, 'bs',markersize=5, label='Acc. Up'  )

# Plot analytical eigenvalues
h_analytical, = ax2.plot(eigenvalues_a.real,eigenvalues_a.imag, 'ko', markerfacecolor='None',markersize=10,label='Analytical')


# Plot eigenvectors
for i in range(4):
    h_va, = ax3.plot(eigenvectors_a[:,2*i],r_a, 'k', linewidth=1, label='Analytical')
    h_vn, = ax3.plot(p_eigenvectors[:,2*i].real/np.max(np.abs(p_eigenvectors[:,2*i].real)),r, 'ko', label='Numerical')


# Eigenvalue plot settings
ax1.set_xlabel('$Re(k_z)$', fontsize=12)
ax1.set_ylabel('$Im(k_z)$', fontsize=12)
ax2.legend(handles=[h_analytical,h_up,h_down],numpoints=1)
ax2.set_xlabel('$Re(k_z)$', fontsize=12)
ax2.set_ylabel('$Im(k_z)$', fontsize=12)

# Eigenvector plot settings
ax3.set(xlim=(-1.,1.), ylim=(0.25,1.0), xlabel="$Re(P_{mn})$", ylabel="Radial coordinate")
ax3.legend(handles=[h_va,h_vn],numpoints=1,loc=0)





# Axis settings
#ax1.legend(handles=[h_analytical,h_up,h_down],numpoints=1)
#ax1.set_xlabel('$Re(k_z)$')
#ax1.set_ylabel('$Im(k_z)$')

plt.show()


