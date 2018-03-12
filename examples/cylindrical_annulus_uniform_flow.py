from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import noisyduck as nd


# Define mean state
gam=1.4; rho=1.2; vr=0.; vt=0.; vz=100.; p=100000.

# Geometry
ri=0.25; ro=1.0; res=50
r = np.linspace(ri,ro,res)

# Define circumferential and temporal wavenumber
omega=3000.; m=2

# Numerical decomposition (raw, filtered)
eigenvalues_r, eigenvectors_rl, eigenvectors_rr = nd.annulus.numerical.decomposition(omega,m,r,rho,vr,vt,vz,p,gam,filter='None',perturb_omega=True)
eigenvalues_f, eigenvectors_fl, eigenvectors_fr = nd.annulus.numerical.decomposition(omega,m,r,rho,vr,vt,vz,p,gam,filter='acoustic',alpha=0.00001,perturb_omega=True)


# Separate eigenvectors into primitive variables
res = len(r)
rho_eigenvectors = eigenvectors_fr[0*res:1*res,:]
vr_eigenvectors  = eigenvectors_fr[1*res:2*res,:]
vt_eigenvectors  = eigenvectors_fr[2*res:3*res,:]
vz_eigenvectors  = eigenvectors_fr[3*res:4*res,:]
p_eigenvectors   = eigenvectors_fr[4*res:5*res,:]


# Nondimensionalize for analytical case
c = np.sqrt(gam*p/rho)
mach = vz/c
omega = omega/c
n = len(eigenvalues_f)

# Analytical decomposition
eigenvalues_a, eigenvectors_a = nd.annulus.analytical.decomposition(omega,m,mach,r,n)





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
for i in range(len(eigenvalues_f)):
    if (eigenvalues_f[i].imag > 0.):
        h_up, = ax2.plot(eigenvalues_f[i].real,eigenvalues_f[i].imag,   'b^',markersize=5, label='Acc. Down')
    elif (eigenvalues_f[i].imag < 0.):
        h_down, = ax2.plot(eigenvalues_f[i].real,eigenvalues_f[i].imag, 'bs',markersize=5, label='Acc. Up'  )

# Plot analytical eigenvalues
h_analytical, = ax2.plot(eigenvalues_a.real,eigenvalues_a.imag, 'ko', markerfacecolor='None',markersize=10,label='Analytical')


# Plot eigenvectors
for i in range(4):
    h_va, = ax3.plot(eigenvectors_a[:,2*i],r, 'k', linewidth=1, label='Analytical')
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

# Display
plt.show()


