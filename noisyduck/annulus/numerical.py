# -*- coding: utf-8 -*-
r"""
Numerical
---------

This module provides a functionality for computing the numerical
eigenvalue/eigenvector decomposition of a uniform axial flow in
an annular cylindrical duct. The decomposition is based on a normal
mode analysis of the three-dimensional linearized Euler equations,
which yields an eigensystem that is discretized and solved numerically.


Theory:
~~~~~~~



Filtering:
~~~~~~~~~~



Example:
~~~~~~~~

::

    eigenvalues, eigenvectors_l, eigenvectors_r =
        noisyduck.annulus.numerical.decomposition(omega,
                                                  m,
                                                  r,
                                                  rho,
                                                  vr,
                                                  vt,
                                                  vz,
                                                  p,
                                                  gam,
                                                  filter='acoustic')

"""
import numpy as np
import scipy
import noisyduck.filter


def decomposition(omega, m, r, rho, vr, vt, vz, p, gam,
                  filter='None', alpha=0.00001, equation='general',
                  perturb_omega=True):
    r""" Compute the numerical eigen-decomposition of the three-dimensional
    linearized Euler equations for a cylindrical annulus.

    Args:
        omega (float): temporal frequency.
        m (int): circumferential wavenumber.
        r (float): array of equally-spaced radius locations for the
                   discretization, including end points.
        rho (float): mean density.
        vr (float): mean radial velocity.
        vt (float): mean tangential velocity.
        vz (float): mean axial velocity.
        p (float): mean pressure.
        gam (float): ratio of specific heats.
        filter (string, optional):
            Optional filter for eigenmodes. values = ['None', 'acoustic']
        alpha (float, optional):
            Criteria governing filtering acoustic modes.
        equation (string, optional):
            Select from of governing equation for the decomposition.
            values = ['general', 'radial equilibrium']
        perturb_omega (bool):
            If true, small imaginary part is added to the temporal
            frequency. Can help in determining direction of propagation.

    Returns:
        (eigenvalues, left_eigenvectors, right_eigenvectors):
            a tuple containing an array of eigenvalues, an array of left
            eigenvectors evaluated at radial locations, an array of right
            eigenvectors evaluated at radial locations.

    Note:
        The eigenvectors being returned include each field
        :math:`[\rho,v_r,v_t,v_z,p]`. The primitive variables can be extracted
        into their own eigenvectors by copying out those entries from the
        returned eigenvectors as:

        ::

            res = len(r)
            rho_eigenvectors = eigenvectors[0*res:1*res,:]
            vr_eigenvectors  = eigenvectors[1*res:2*res,:]
            vt_eigenvectors  = eigenvectors[2*res:3*res,:]
            vz_eigenvectors  = eigenvectors[3*res:4*res,:]
            p_eigenvectors   = eigenvectors[4*res:5*res,:]

    """
    res = len(r)

    # Construct eigensystem
    if (equation == 'general'):
        M, N = construct_numerical_eigensystem_general(
            omega, m, r, rho, vr, vt, vz, p, gam, perturb_omega)
    else:
        raise ValueError("Invalid input for 'equation'. Valid options are 'general'")


    # Solve Standard Eigenvalue Problem for complex, nonhermitian system
    evals, evecs_l, evecs_r = scipy.linalg.eig(np.matmul(np.linalg.inv(N), M),
                                               left=True,
                                               right=True,
                                               overwrite_a=True,
                                               overwrite_b=True)

    # Add radial velocity end points back where they were removed due to
    # boundary conditions
    evecs_r = np.insert(evecs_r, [res], [0.], axis=0)
    evecs_l = np.insert(evecs_l, [res], [0.], axis=0)
    evecs_r = np.insert(evecs_r, [2*res-1], [0.], axis=0)
    evecs_l = np.insert(evecs_l, [2*res-1], [0.], axis=0)

    # Filtering
    if (filter == 'acoustic'):
        evals, evecs_l, evecs_r = noisyduck.filter.physical(evals,
                                                            evecs_l,
                                                            evecs_r,
                                                            r,
                                                            alpha_cutoff=alpha,
                                                            filters=filter)

    # Return conventional definition of the left eigenvector
    evecs_l = np.copy(evecs_l.conj())

    return evals, evecs_l, evecs_r





def construct_numerical_eigensystem_general(
        omega, m, r, rho, vr, vt, vz, p, gam, perturb_omega=True):
    r""" Constructs the numerical representation of the eigenvalue problem
    associated with the three-dimensional linearized euler equations subjected
    to a normal mode analysis.

    NOTE: If perturb_omega=True, a small imaginary part is added to the
    temporal frequency to facilitate determining the propagation direction
    of eigenmodes based on the sign of the imaginary part of their eigenvalue.
    That is: :math:`\omega = \omega - 10^{-5}\omega j`.
    See Moinier and Giles[2].

    [1] Kousen, K. A., "Eigenmodes of Ducted Flows With Radially-Dependent
        Axial and Swirl Velocity Components", NASA/CR 1999-208881, March 1999.
    [2] Moinier, P., and Giles, M. B., "Eigenmode Analysis for Turbomachinery
        Applications", Journal of Propulsion and Power, Vol. 21, No. 6,
        November-December 2005.

    Args:
        omega (float): temporal frequency.
        m (int): circumferential wavenumber.
        r (float): array of equally-spaced radius locations for the
                   discretization, including end points.
        rho (float): mean density.
        vr (float): mean radial velocity.
        vt (float): mean tangential velocity.
        vz (float): mean axial velocity.
        p (float): mean pressure.
        gam (float): ratio of specific heats.
        perturb_omega (bool): If true, small imaginary part is added to the
                              temporal frequency. Can help in determining
                              direction of propagation.

    Returns:
        (M, N): left-hand side of generalized eigenvalue problem, right-hand
                side of generalized eigenvalue problem.
    """

    # Define real/imag parts for temporal frequency
    romega = omega
    if (perturb_omega):
        iomega = 10.e-5*romega
    else:
        iomega = 0.

    # Define geometry and discretization
    res = len(r)
    ri = np.min(r)
    ro = np.max(r)
    dr = (ro-ri)/(res-1)
    nfields = 5
    dof = res*nfields

    # Check if input mean quantities are scalar.
    # If so, expand them to a vector
    if (type(rho) is float):
        rho = np.full(res, rho)
    if (type(vr) is float):
        vr = np.full(res, vr)
    if (type(vt) is float):
        vt = np.full(res, vt)
    if (type(vz) is float):
        vz = np.full(res, vz)
    if (type(p) is float):
        p = np.full(res, p)

    # Allocate storage
    M = np.zeros([dof, dof], dtype=np.complex)
    N = np.zeros([dof, dof], dtype=np.complex)

    # Submatrices for discretization
    stencil   = np.zeros([res, res])
    identity  = np.zeros([res, res])
    ridentity = np.zeros([res, res])

    # Construct fourth-order finite difference stencil
    #
    # Stencil operations
    #
    #   stencil*f
    #       => d(f u')/dr
    #       => returns matrix operator, L corresponding to
    #          d(f*u')/dr, where Lu' = d(f*u')/dr
    #
    #   np.matmul(stencil,f)
    #       => d(f)/dr
    #       => returns vector of evaluated derivatives, d(f)/dr
    #
    #   np.matmul(identity*f,stencil)
    #       => f d(u')/dr
    #       => returns matrix operator, L corresponding to
    #          (f d()/dr), where Lu' = f*d(u')/dr
    #
    stencil[0, 0:5] = [-25., 48., -36, 16., -3.]
    stencil[1, 0:5] = [-3., -10., 18., -6., 1.]
    stencil[res-2, res-5:res] = [-1., 6., -18., 10., 3.]
    stencil[res-1, res-5:res] = [3., -16., 36., -48., 25.]
    for i in range(2, res-2):
        stencil[i, i-2] = 1.
        stencil[i, i-1] = -8.
        stencil[i, i+1] = 8.
        stencil[i, i+2] = -1.
    stencil = (1./(12.*dr))*stencil

    # Construct identity matrix for source terms
    for i in range(res):
        identity[i, i] = 1.

    # Construct identity scaled by 1/r
    for i in range(res):
        ridentity[i, i] = 1./r[i]

    # Compute radial derivatives
    drho_dr = np.matmul(stencil, rho)
    dvr_dr  = np.matmul(stencil, vr)
    dvt_dr  = np.matmul(stencil, vt)
    dvz_dr  = np.matmul(stencil, vz)
    dp_dr   = np.matmul(stencil, p)

    # Index Legend:
    # irs = irow_start
    # ics = icol_start

    # Block 1,1
    irow = 1
    icol = 1
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (
        1j*identity*romega +                       # temporal source(real)
        (identity*iomega) +                        # temporal source(imag)
        np.matmul(identity*vr, stencil) +          # bar{A} radial
        1j*ridentity*float(m)*vt)                  # bar{B} circum. source

    N[irs:irs + res, ics:ics + res] += 1j*identity*vz   # axial source

    # Block 2,2
    irow = 2
    icol = 2
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (
        1j*identity*romega +                        # temporal source(real)
        (identity*iomega) +                         # temporal source(imag)
        np.matmul(identity*vr, stencil) +           # bar{A} rad derivative
        1j*ridentity*float(m)*vt )                  # bar{B} circ source

    N[irs:irs + res, ics:ics + res] += 1j*identity*vz   # bar{A} axial source

    # Block 3,3
    irow = 3
    icol = 3
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (
        1j*identity*romega +                        # temporal source(real)
        (identity*iomega) +                         # temporal source(imag)
        np.matmul(identity*vr, stencil) +           # bar{A} rad derivative
        1j*ridentity*float(m)*vt)                   # bar{B} circ source

    N[irs:irs + res, ics:ics + res] += 1j*identity*vz   # bar{A} axial source

    # Block 4,4
    irow = 4
    icol = 4
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (
        1j*identity*romega +                        # temporal source(real)
        (identity*iomega) +                         # temporal source(imag)
        np.matmul(identity*vr, stencil) +           # bar{A} rad derivative
        1j*ridentity*float(m)*vt )                  # bar{B} circ source

    N[irs:irs + res, ics:ics + res] += 1j*identity*vz   # bar{A} axial source

    # Block 5,5
    irow = 5
    icol = 5
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (
        1j*identity*romega +                        # temporal source(real)
        (identity*iomega) +                         # temporal source(imag)
        np.matmul(identity*vr, stencil) +           # bar{A} rad derivative
        1j*ridentity*float(m)*vt)                   # bar{C} circ source

    N[irs:irs + res, ics:ics + res] += 1j*identity*vz   # bar{A} axial source

    # Block 2,1
    irow = 2
    icol = 1
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (
        (-ridentity*vt*vt/rho) )                    # bar{D} eqn/coord source

    # Block 3,1
    irow = 3
    icol = 1
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    # NO CONTRIBUTIONS

    # Block 4,1
    irow = 4
    icol = 1
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    # NO CONTRIBUTIONS

    # Block 1,2
    irow = 1
    icol = 2
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (
        np.matmul(identity*rho, stencil) +          # bar{A} 
        ridentity*rho +                             # bar{D} 
        identity*drho_dr )                          # bar{D}

    # Block 3,2
    irow = 3
    icol = 2
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (
        ridentity*vt +                              # bar{D} 
        identity*dvt_dr )                           # bar{D}

    # Block 4,2
    irow = 4
    icol = 2
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += identity*dvz_dr  # bar{D}

    # Block 5,2
    irow = 5
    icol = 2
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (
        np.matmul(identity*(p*gam), stencil) +      # bar{A} 
        ridentity*p*gam +                           # bar{D} 
        identity*dp_dr )                            # bar{D}

    # Block 1,3
    irow = 1
    icol = 3
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += 1j*ridentity*rho*float(m)    # bar{B}

    # Block 2,3
    irow = 2
    icol = 3
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += (-ridentity*2.*vt)   # bar{D}

    # Block 5,3
    irow = 5
    icol = 3
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += 1j*ridentity*gam*p*float(m)  # bar{B}

    # Block 1,4
    irow = 1
    icol = 4
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    N[irs:irs + res, ics:ics + res] += 1j*identity*rho  # bar{C} axial source

    # Block 5,4
    irow = 5
    icol = 4
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    N[irs:irs + res, ics:ics + res] += 1j*identity*gam*p  # bar{C} axial source

    # Block 2,5
    irow = 2
    icol = 5
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] += np.matmul(identity*(1./rho), stencil) # bar{A}

    # Block 3,5
    irow = 3
    icol = 5
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    M[irs:irs + res, ics:ics + res] = 1j*ridentity*float(m)/rho # bar{B}

    # Block 4,5
    irow = 4
    icol = 5
    irs = 0 + res*(irow-1)
    ics = 0 + res*(icol-1)
    N[irs:irs + res, ics:ics + res] += 1j*identity/rho  # bar{C} axial source

    # Remove rows/columns due to solid wall boundary condition: no velocity
    # normal to boundary, assumes here wall normal vector is aligned with
    # radial coordinate.
    M = np.delete(M, (res, 2*res-1), axis=0)
    M = np.delete(M, (res, 2*res-1), axis=1)

    N = np.delete(N, (res, 2*res-1), axis=0)
    N = np.delete(N, (res, 2*res-1), axis=1)

    # Move N to right-hand side
    N = np.copy(-N)

    return M, N




