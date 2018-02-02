import numpy as np
import scipy







def eigen_decomposition(m,omega,ri,ro,rho,u,v,w,p,gam):
    """ Compute the numerical eigen-decomposition of the three-dimensional linearized
    Euler equations for a cylindrical annulus.

    Args:
        m: circumferential wavenumber
        n: number of radial eigenmodes to retain
        omega: temporal frequency
        ri: inner radius
        ro: outer radius

    """


    # Define real/imag parts for temporal frequency
    romega = omega
    iomega = -10.e-5*romega


    # Define geometry and discretization
    res = 50
    dr = (ro-ri)/(res-1)
    nfields = 5
    r = np.linspace(ri,ro,res)
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
    
    # Solve Generalized Eigenvalue Problem for complex, nonhermitian system
    eigenvalues, eigenvectors = scipy.linalg.eig(M,N,right=True,overwrite_a=True,overwrite_b=True)
    
    # Add radial velocity end points back
    eigenvectors = np.insert(eigenvectors, [res]    , [0.] ,axis=0)
    eigenvectors = np.insert(eigenvectors, [2*res-1], [0.] ,axis=0)
    
    return eigenvalues, eigenvectors


















