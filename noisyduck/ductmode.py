from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib         import rc, rcParams
from annulus_analytical import compute_eigenvalues, compute_eigenvector
from annulus_numerical  import eigen_decomposition
from filter             import filter_drp15



# Font/plot setup
rc('text', usetex=True)
rc('font', family='serif')
rcParams.update({'figure.autolayout': True})
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig1.set_facecolor('White')
fig2.set_facecolor('White')
fig3.set_facecolor('White')
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)



# Define mean state
gam = 1.4
rho = 1.2212179
u   = 0.
v   = 0.
w   = 103.2586448
p   = 103341.6659



# Define circumferential and temporal wavenumber
m     = 2
omega = 3441.9548



# Define geometry and discretization
res = 50
ri = 0.25
ro = 1.0
dr = (ro-ri)/(res-1)
nfields = 5
r = np.linspace(ri,ro,res)
dof = res*nfields



# Compute eigen-decomposition
eigenvalues, eigenvectors = eigen_decomposition(m,omega,ri,ro,rho,u,v,w,p,gam)



# Separate eigenvectors into primitive variables
rho_eigenvectors = eigenvectors[0*res:1*res,:]
u_eigenvectors   = eigenvectors[1*res:2*res,:]
v_eigenvectors   = eigenvectors[2*res:3*res,:]
w_eigenvectors   = eigenvectors[3*res:4*res,:]
p_eigenvectors   = eigenvectors[4*res:5*res,:]



# Create separate copy so we can compare before/after modes
pr_eigenvectors   = np.copy(np.real(p_eigenvectors))
pr_eigenvectors_f = np.copy(pr_eigenvectors)



# Call filter for each eigenvector
for i in range(pr_eigenvectors.shape[1]):
    pr_eigenvectors_f[:,i] = filter_drp15(pr_eigenvectors[:,i])

# Select low-wavenumber modes based on the ratio of their vector norms
amp   = np.zeros(pr_eigenvectors.shape[1])
amp_f = np.zeros(pr_eigenvectors.shape[1])
for i in range(pr_eigenvectors.shape[1]):
    for j in range(len(r)-1):
        amp[i]   = amp[i]   + 0.5*(r[j+1]-r[j])*(r[j]*pr_eigenvectors[  j,i]*pr_eigenvectors[  j,i]  +  r[j+1]*pr_eigenvectors[  j+1,i]*pr_eigenvectors[  j+1,i])
        amp_f[i] = amp_f[i] + 0.5*(r[j+1]-r[j])*(r[j]*pr_eigenvectors_f[j,i]*pr_eigenvectors_f[j,i]  +  r[j+1]*pr_eigenvectors_f[j+1,i]*pr_eigenvectors_f[j+1,i])

# Compute selection criteria
alpha = amp_f/amp

# Plot vectors that satisfy criteria
for i in range(len(alpha)):
    if (alpha[i] < 0.00001):     # res = 50
        if (np.imag(eigenvalues[i]) > 0.):
            h_up, = ax1.plot(np.real(eigenvalues[i]),np.imag(eigenvalues[i]),   'b^',markersize=6, label='Acc. Down')
        elif (np.imag(eigenvalues[i]) < 0.):
            h_down, = ax1.plot(np.real(eigenvalues[i]),np.imag(eigenvalues[i]), 'bs',markersize=6, label='Acc. Up'  )
        ax2.plot(pr_eigenvectors[:,i]/max(abs(pr_eigenvectors[:,i])), r, '-')






# Good to compare against aiaaj_iftn_2009 Fig 13
#m  = 10
#Ri = 0.52
#Ro = 1.0

mach = 0.3
omega = 10
sigma = ri





# Get eigenvalues for each radial mode
nrmodes = 10
alpha = np.zeros(nrmodes)
alpha = compute_eigenvalues(m,nrmodes,ri,ro)

# Compute axial wave numbers
k = np.zeros(2*len(alpha),dtype=np.complex)
for i in range(len(alpha)):
    a = mach*mach - 1.
    b = 2.*omega*mach
    c = omega*omega - alpha[i]*alpha[i]
    roots = np.roots([a,b,c])
    k[2*i]   = roots[0]
    k[2*i+1] = roots[1]
h_analytical, = ax1.plot(k.real,k.imag, 'o', markerfacecolor='None',markersize=10,label='Analytical')




# Compute radius and percent span location
r_array = np.linspace(ri,ro,100)
p_span  = np.zeros(100)
for i in range(len(r_array)):
    p_span[i] = 100.*(r_array[i]-ri)/(ro-ri)

# Loop through the radial modes 
for rmode in range(nrmodes):
    eigenvector = compute_eigenvector(r_array,sigma,m,alpha[rmode])
    ax3.plot(eigenvector,p_span)
    




ax1.legend(handles=[h_analytical,h_up,h_down],numpoints=1)
ax1.set_xlabel('$Re(k_z)$')
ax1.set_ylabel('$Im(k_z)$')
#ax1.set_xlim((-15.,10.))
#ax1.set_ylim((0.6,1.0))
ax3.set_ylim((0,100))
ax3.set_xlim((-1.1,1.1))
ax3.set_xlabel('Radial mode amplitude', weight='bold', fontsize=12, labelpad=10)
ax3.set_ylabel('$\%$ Span',             weight='bold', fontsize=12, labelpad=10)


#fig.set_size_inches(5,8)
#fig.savefig('radial_modes_1d.png', bbox_inches='tight')
plt.show()



