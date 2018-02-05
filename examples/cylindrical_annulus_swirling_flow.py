from __future__ import division
import pylab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import noisyduck as nd

# Geometry
res   = 50
ri    = 0.4
ro    = 1.0
r = np.linspace(ri,ro,res)

# Define mean state
GAMMA = 0.2
gam = 1.4
u   = 0.
v   = GAMMA/r
w   = 0.3
p   = (1./gam) + (GAMMA*GAMMA/2.)*(1. - 1./(r*r))

# Constant density: Tam, Auriault
#rho = 1.0
# Homentropic: Kousen, Nijbour
rho = (1. + GAMMA*GAMMA*(gam-1.)*(1. - 1./(r*r))/2.)**(1./(gam-1.))


# Define circumferential and temporal wavenumber
omega = -10.
m = 2


# Numerical decomposition
eigenvalues_r, eigenvectors_r = nd.annulus.numerical.decomposition(omega,m,r,rho,u,v,w,p,gam,filter='None')
eigenvalues_n, eigenvectors_n = nd.annulus.numerical.decomposition(omega,m,r,rho,u,v,w,p,gam,filter='acoustic',alpha=0.00001)

# Separate eigenvectors into primitive variables
res = len(r)
rho_eigenvectors = eigenvectors_n[0*res:1*res,:]
u_eigenvectors   = eigenvectors_n[1*res:2*res,:]
v_eigenvectors   = eigenvectors_n[2*res:3*res,:]
w_eigenvectors   = eigenvectors_n[3*res:4*res,:]
p_eigenvectors   = eigenvectors_n[4*res:5*res,:]


# Eigenvalues for free vortex swirling flow in a cylindrical annulus.
# 
# Data Extracted from:
# Nijbour, "Eigenvalues and Eigenfunctions of Ducted Swirling Flows", 7th AIAA/CEAS Aeroacoustics Conference, AIAA 2001-2178.
# 
# Flow description:
#     u_r     = 0
#     u_theta = G/r
#     u_z     = M = 0.3
#     G = 0.2
#     gam = 1.4
# 
#     Homentropic flow, density consistent with Kousen
#     rho = (1 + G*G*(gam-1.)*(1. - 1./r*r)/2)**(1./(gam-1.))
# 
#     Pressure
#     p = (1./gam) - (G*G/2.)*(1. - 1./r*r)
# 
#     Temporal, circumferential wavenumbers:
#     omega = -10.
#     m     = 2
# 
# Re{eigenvalue}, Im{eigenvalue}
eigenvalues_nijbour = np.array([[-3.027340779492441,  48.36867862969005  ],
                                [-3.061040170507461,  42.82218597063621  ],
                                [-3.062863166930221,  37.11256117455139  ],
                                [-3.0646340777409034, 31.56606851549755  ],
                                [-3.066509159775743,  25.693311582381725 ],
                                [-3.0683842418105804, 19.8205546492659   ],
                                [-3.0703634950695786, 13.621533442088094 ],
                                [-3.008902472816523,  6.117455138662315  ],
                                [-3.0128609793345174,-6.280587275693314  ],
                                [-3.0471333120824085,-13.621533442088094 ],
                                [-3.0810931311578234,-19.9836867862969   ],
                                [-3.0830202988047404,-26.019575856443723 ],
                                [-3.0848432952275022,-31.729200652528547 ],
                                [-3.054737811445923, -37.438825448613386 ],
                                [-3.056508722256604, -42.98531810766721  ],
                                [-3.058331718679364, -48.694942903752036 ],
                                [-9.939361930417792, -0.08156606851550663],
                                [-12.940639069625966,-0.08156606851550663],
                                [3.9814554386754395, -0.08156606851550663],
                                [6.695376256044533,  -0.08156606851550663]])



# Font/plot setup
rc('text', usetex=True)
rc('font', family='serif')
rcParams.update({'figure.autolayout': True})
fig1 = plt.figure(figsize=(4.5,6))
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

# Plot filtered numerical eigenvalues
for i in range(len(eigenvalues_n)):
    if (eigenvalues_n[i].imag > 0.):
        h_up, = ax2.plot(eigenvalues_n[i].real,eigenvalues_n[i].imag,   'b^',markersize=5, label='Acc. Down')
    elif (eigenvalues_n[i].imag < 0.):
        h_down, = ax2.plot(eigenvalues_n[i].real,eigenvalues_n[i].imag, 'bs',markersize=5, label='Acc. Up'  )

# Plot analytical eigenvalues
h_analytical, = ax2.plot(eigenvalues_nijbour[:,0],eigenvalues_nijbour[:,1], 'ko', markerfacecolor='None',markersize=10,label='Nijbour(2001)')



# Plot first 4 eigenvectors
for i in range(4):
    ax3.plot(p_eigenvectors[:,2*i].real/np.max(np.abs(p_eigenvectors[:,2*i].real)),r, 'ko')


# Eigenvalue plot settings
ax1.set_xlabel('$Re(k_z)$', fontsize=12)
ax1.set_ylabel('$Im(k_z)$', fontsize=12)
ax2.legend(handles=[h_analytical,h_up,h_down],numpoints=1)
ax2.set_xlabel('$Re(k_z)$', fontsize=12)
ax2.set_ylabel('$Im(k_z)$', fontsize=12)

# Eigenvector plot settings
ax3.set(xlim=(-1.,1.), ylim=(0.4,1.0), xlabel="$Re(P_{mn})$", ylabel="Radial coordinate")


#fig1.savefig('annulus_swirlingflow_eigenvalues_raw.png')
#fig2.savefig('annulus_swirlingflow_eigenvalues_filtered.png')
#fig3.savefig('annulus_swirlingflow_eigenvectors.png')
plt.show()


