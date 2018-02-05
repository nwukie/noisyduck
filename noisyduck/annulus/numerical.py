# -*- coding: utf-8 -*-
r""" 
Numerical
---------

This module provides a functionality for computing the numerical eigenvalue/eigenvector
decomposition of a uniform axial flow in an annular cylindrical duct. The decomposition
is based on a normal mode analysis of the three-dimensional linearized Euler equations,
which yields an eigensystem that is discretized and solved numerically.


Theory:
~~~~~~~



Filtering:
~~~~~~~~~~



Example:
~~~~~~~~

::
    
    eigenvalues, eigenvectors, r = noisyduck.annulus.numerical.decomposition(omega,m,ri,ro,rho,u,v,w,p,gam)

"""


import numpy as np
import scipy
import noisyduck.filter






def decomposition(omega,m,r,rho,u,v,w,p,gam,filter='None',alpha=0.00001):
    r""" Compute the numerical eigen-decomposition of the three-dimensional linearized
    Euler equations for a cylindrical annulus.

    Args:
        omega (float): temporal frequency.
        m (int): circumferential wavenumber.
        r (float): array of equally-spaced radius locations for the discretization, including end points.
        rho (float): mean density.
        u (float): mean radial velocity.
        v (float): mean tangential velocity.
        w (float): mean axial velocity.
        p (float): mean pressure.
        gam (float): ratio of specific heats.
        filter (string, optional): Optional filter for eigenmodes. allowable values = ['None', 'acoustic']
        alpha (float, optional): Criteria governing filtering acoustic modes. 

    Returns:
        (eigenvalues, eigenvectors, r): a tuple containing an array of eigenvalues, an array 
        of eigenvectors evaluated at radial locations, and an array of those radial locations.

    Note:
        The eigenvectors being returned include each field :math:`[\rho,u,v,w,p]`. The primitive
        variables can be extracted into their own eigenvectors by copying out those entries from 
        the returned eigenvectors as:

        ::

            res = len(r)
            rho_eigenvectors = eigenvectors[0*res:1*res,:]
            u_eigenvectors   = eigenvectors[1*res:2*res,:]
            v_eigenvectors   = eigenvectors[2*res:3*res,:]
            w_eigenvectors   = eigenvectors[3*res:4*res,:]
            p_eigenvectors   = eigenvectors[4*res:5*res,:]

    """
    res = len(r)
    ri = np.min(r)
    ro = np.max(r)


    # Construct eigensystem
    M, N = construct_numerical_eigensystem(omega,m,r,rho,u,v,w,p,gam)

    # Solve Generalized Eigenvalue Problem for complex, nonhermitian system
    eigenvalues, eigenvectors = scipy.linalg.eig(M,N,right=True,overwrite_a=True,overwrite_b=True)
    
    # Add radial velocity end points back where they were removed due to boundary conditions
    eigenvectors = np.insert(eigenvectors, [res]    , [0.] ,axis=0)
    eigenvectors = np.insert(eigenvectors, [2*res-1], [0.] ,axis=0)




    if (filter == 'acoustic'):
        eigenvalues, eigenvectors = noisyduck.filter.physical(eigenvalues,eigenvectors,r,alpha_cutoff=alpha,filters=filter)


    return eigenvalues, eigenvectors


    


def construct_numerical_eigensystem(omega,m,r,rho,u,v,w,p,gam):
    """ Constructs the numerical representation of the eigenvalue problem associated
    with the three-dimensional linearized euler equations subjected to a normal mode
    analysis.

    NOTE: a small imaginary part is added to the temporal frequency to facilitate 
    determining the propagation direction of eigenmodes based on the sign of the 
    imaginary part of their eigenvalue.

    Args:
        omega (float): temporal frequency.
        m (int): circumferential wavenumber.
        r (float): array of equally-spaced radius locations for the discretization, including end points.
        rho (float): mean density.
        u (float): mean radial velocity.
        v (float): mean tangential velocity.
        w (float): mean axial velocity.
        p (float): mean pressure.
        gam (float): ratio of specific heats.

    Returns:
        (M, N, r): left-hand side of generalized eigenvalue problem, right-hand side 
        of generalized eigenvalue problem, and radial coordinates of the discretization.
    """

    # Define real/imag parts for temporal frequency
    romega = omega
    iomega = -10.e-5*romega

    # Define geometry and discretization
    res = len(r)
    ri = np.min(r)
    ro = np.max(r)
    dr = (ro-ri)/(res-1)
    nfields = 5
    dof = res*nfields


    # Allocate storage
    M  = np.zeros([dof,dof], dtype=np.complex)
    N  = np.zeros([dof,dof], dtype=np.complex)
    vl = np.zeros([dof,dof], dtype=np.complex)
    vr = np.zeros([dof,dof], dtype=np.complex)
    
    # Submatrices for discretization
    stencil     = np.zeros([res,res])
    identity    = np.zeros([res,res])
    ridentity   = np.zeros([res,res])
    zero_matrix = np.zeros([res,res])

    # Construct fourth-order finite difference stencil
    stencil[0,0:5] = [-25., 48., -36, 16., -3.]
    stencil[1,0:5] = [-3., -10., 18., -6., 1.]
    stencil[res-2, res-5:res] = [-1., 6., -18., 10., 3.]
    stencil[res-1, res-5:res] = [3., -16., 36., -48., 25.]
    for i in range(2,res-2):
        stencil[i,i-2] =  1.
        stencil[i,i-1] = -8.
        stencil[i,i+1] =  8.
        stencil[i,i+2] = -1.
    # Scale by dr
    stencil = (1./(12.*dr))*stencil
    
    
    # Construct identity matrix for source terms
    for i in range(res):
        identity[i,i] = 1.
    
    
    # Construct identity scaled by 1/r
    for i in range(res):
        ridentity[i,i] = 1./r[i]
    



    # Block 1,1
    irow = 1
    icol = 1
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*romega        # temporal source(real)
                                                                        -    identity*iomega        # temporal source(imag)
                                                                        +    stencil*u              # radial derivative
                                                                        +    ridentity*u            # radial source
                                                                        + 1j*ridentity*float(m)*v ) # circumferential source

    N[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( N[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*w )           # axial source

    # Block 2,2
    irow = 2
    icol = 2
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*romega        # temporal source(real)
                                                                        -    identity*iomega        # temporal source(imag)
                                                                        +    stencil*u              # radial derivative
                                                                        +    ridentity*u            # radial source
                                                                        + 1j*ridentity*float(m)*v ) # circumferential source

    N[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( N[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*w )           # axial source


    # Block 3,3
    irow = 3
    icol = 3
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*romega        # temporal source(real)
                                                                        -    identity*iomega        # temporal source(imag)
                                                                        +    stencil*u              # radial derivative
                                                                        +    ridentity*u            # radial source
                                                                        + 1j*ridentity*float(m)*v   # circumferential source
                                                                        +    ridentity*u )          # equation/coordinate source

    N[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( N[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*w )           # axial source



    # Block 4,4
    irow = 4
    icol = 4
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*romega        # temporal source(real)
                                                                        -    identity*iomega        # temporal source(imag)
                                                                        +    stencil*u              # radial derivative
                                                                        +    ridentity*u            # radial source
                                                                        + 1j*ridentity*float(m)*v ) # circumferential source

    N[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( N[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*w )           # axial source


    # Block 5,5
    irow = 5
    icol = 5
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*romega        # temporal source(real)
                                                                        -    identity*iomega        # temporal source(imag)
                                                                        +    stencil*u              # radial derivative
                                                                        +    ridentity*u            # radial source
                                                                        + 1j*ridentity*float(m)*v   # circumferential source
                                                                        +    ridentity*(gam-1.)*u ) # equation/coordinate source

    N[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( N[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*w )           # axial source


    # Block 2,1
    irow = 2
    icol = 1
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        - ridentity*v*v/rho )   # equation/coordinate source

    # Block 3,1
    irow = 3
    icol = 1
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + ridentity*u*v/rho )   # equation/coordinate source


    # Block 1,2
    irow = 1
    icol = 2
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + stencil*rho           # radial derivative
                                                                        + ridentity*rho )       # radial source

    # Block 3,2
    irow = 3
    icol = 2
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + ridentity*v )         # equation/coordinate source

    # Block 5,2
    irow = 5
    icol = 2
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + stencil*gam*p                 # radial derivative
                                                                        + ridentity*gam*p               # radial source
                                                                        - ridentity*(gam-1.)*rho*v*v )  # equation/coordinate source

    # Block 1,3
    irow = 1
    icol = 3
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*ridentity*rho*float(m) )   # circumferential source

    # Block 2,3
    irow = 2
    icol = 3
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        - ridentity*2.*v )              # equation/coordinate source

    # Block 5,3
    irow = 5
    icol = 3
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*ridentity*gam*p*float(m)       # circumferential source
                                                                        +    ridentity*(gam-1.)*rho*u*v )   # equation/coordinate source

    # Block 1,4
    irow = 1
    icol = 4
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    N[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( N[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*rho )                # axial source

    # Block 5,4
    irow = 5
    icol = 4
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    N[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( N[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity*gam*p )              # axial source

    # Block 2,5
    irow = 2
    icol = 5
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + stencil/rho                   # radial derivative
                                                                        + ridentity/rho                 # radial source
                                                                        - ridentity/rho )               # equation/coordinate source
    # Block 3,5
    irow = 3
    icol = 5
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    M[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( M[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*ridentity*float(m)/rho )   # circumferential source

    # Block 4,5
    irow = 4
    icol = 5
    irow_start = 0 + res*(irow-1)
    icol_start = 0 + res*(icol-1)
    N[irow_start:irow_start + (res), icol_start:icol_start + (res)] = ( N[irow_start:irow_start+(res),icol_start:icol_start+(res)] 
                                                                        + 1j*identity/rho )             # axial source



    # Remove rows/columns due to solid wall boundary condition: no velocity normal to boundary, assumes 
    # here wall normal vector is aligned with radial coordinate.
    M = np.delete(M, (res,2*res-1), axis=0)
    M = np.delete(M, (res,2*res-1), axis=1)
    
    N = np.delete(N, (res,2*res-1), axis=0)
    N = np.delete(N, (res,2*res-1), axis=1)
    
    
    # Move N to right-hand side
    N = -N


    return M, N





