# -*- coding: utf-8 -*-
r""" 
Analytical
----------

This module provides a functionality for computing the eigenvalue/eigenvector
decomposition of a uniform axial flow in an annular cylindrical duct. The decomposition
is based on an analytical solution of the convected wave equation for pressure, 
yielding the acoustic part of the eigen decomposition.


Theory:
~~~~~~~

The analysis here follows that of Moinier and Giles, "Eigenmode Analysis for Turbomachinery Applications", Journal of Propulsion and Power, Vol. 21, No. 6, 2005.

For a uniform axial mean, linear pressure perturbations satisfy the convected wave equation

.. math::

    \left( \frac{\partial}{\partial t} + M \frac{\partial}{\partial z} \right)^2 p = \nabla^2 p \quad\quad\quad \lambda < r < 1


where :math:`M` is the Mach number of the axial mean flow and :math:`r` has been normalized
by the outer duct radius. Boundary conditions for a hardwalled duct imply zero radial 
velocity, which is imposed using the condition


.. math::

    \frac{\partial p}{\partial r} = 0   \quad\quad \text{at}    \quad\quad r=\lambda,1


Conducting a normal mode analysis of the governing equation involves assuming a solution
with a form 

.. math::

    p(r,\theta,z,t) = e^{j(\omega t + m \theta + k z)}P(r)

This leads to a Bessel equation

.. math::

    \frac{1}{r} \frac{d}{dr} \left(r \frac{dP}{dr} \right) + \left( \mu^2 - \frac{m^2}{r^2} \right)P = 0    \quad\quad \lambda < r < 1

where :math:`\mu^2 = (\omega + M k)^2 - k^2`. The solution to the Bessel equation is

.. math::

    P(r) = a J_m(\mu r) + b Y_m(\mu r)

where :math:`J_m` and :math:`Y_m` are Bessel functions. Applying boundary conditions
to the general solution gives a set of two equations

.. math::

    \begin{bmatrix}
        J'_m(\mu \lambda)   &   Y'_m(\mu \lambda) \\
        J'_m(\mu)           &   Y'_m(\mu)
    \end{bmatrix}
    \begin{bmatrix}
        a \\ b
    \end{bmatrix}
    = 0

which has nontrivial solutions as long as the determinant is zero. Solving for the 
zeros of the determinant give :math:`\mu`, at which point the quadratic equation above
can be solved for the axial wavenumbers :math:`k`.

Example:
~~~~~~~~

::
    
    eigenvalues, eigenvectors, r = noisyduck.annulus.analytical.decomposition(omega,m,mach,ri,ro,n)

"""
import numpy as np
import scipy.special as sp
import scipy.optimize as op



def decomposition(omega,m,mach,ri,ro,n):
    """ This procedure computes the analytical eigen-decomposition of 
    the convected wave equation.

    Args:
    omega (float): temporal wave number.
    m (int): circumferential wave number.
    mach (float): Mach number.
    ri (float): inner radius.
    ro (float): outer radius.
    n (int): number of eigenvalues/eigenvectors to compute.

    Returns:
        (eigenvalues, eigenvectors, r): a tuple containing an array of eigenvalues, an array of eigenvectors evaluated at radial locations, and an array of those radial locations.

    """

    eigenvalues = compute_eigenvalues(omega,m,mach,ri,ro,n)

    r = np.linspace(ri,ro,100)
    eigenvectors = np.zeros((100,n))
    for i in range(n):
        eigenvectors[:,i] = compute_eigenvector(r,(ri/ro),m,omega)

    return eigenvalues, eigenvectors, r




def eigensystem(b,m,ri,ro):
    r""" Computes the function associated with the eigensystem of the 
    convected wave equation. The location of the zeros for this function 
    correspond to the eigenvalues for the convected wave equation.

    .. math::

        f = J_m(b*ri)*Y_m(b*ro)  -  J_m(b*ro)*Y_m(b*ri) 

    Args:
        b (float): coordinate.
        m (int): circumferential wave number.
        ri (float): inner radius of a circular annulus.
        ro (float): outer radius of a circular annulus.

    """
    #if (b > 200.):
    #    print "WARNING: function 'b_eig_fcn' out of range"
    f = sp.jvp(m,b*ri)*sp.yvp(m,b*ro) - sp.jvp(m,b*ro)*sp.yvp(m,b*ri)
    return f



def compute_zeros(m,mach,ri,ro,n):
    """ This procedure compute the zeros of the determinant for the convected
    wave equation. 
    
    A uniform set of nodes is created to search for sign changes
    in the value of the function. When a sign change is detected, it is known
    that a zero is close. The function is then passed to a bisection routine
    to find the location of the zero. 

    Args:
        m (int): circumferential wavenumber.
        mach (float): Mach number
        ri (float): inner radius of a circular annulus.
        ro (float): outer radius of a circular annulus.
        n (int): number of eigenvalues to compute.

    Returns:
        An array of the first 'n' eigenvalues for the system.

    """
    zero_loc = np.zeros(n)
    for rmode in range(n):
        if rmode == 0:
            beta = np.linspace(0.5,200,2000)
        else:
            beta = np.linspace(zero_loc[rmode-1]+0.1,200,20000)

        for i in range(len(beta)):
            f = eigensystem(beta[i],m,ri,ro)
            fnew = f
            bnew = beta[i]
            if i > 0 :
                #Test for sign change
                if np.sign(fnew) != np.sign(fold):
                    #Sign changed, so start bisection
                    zero_loc[rmode] = op.bisect(eigensystem,bold,bnew,(m,ri,ro))
                    break
                else:
                    fold = fnew
                    bold = bnew
            else:
                fold = fnew
                bold = bnew

    return zero_loc









def compute_eigenvalues(omega,m,mach,ri,ro,n):
    """ This procedure compute the analytical eigenvalues for the convected
    wave equation. A uniform set of nodes is created to search for sign changes
    in the value for the eigensystem. When a sign change is detected, it is known
    that a zero is close. The eigensystem is then passed to a bisection routine
    to find the location of the zero. The location corresponds to the eigenvalue
    for the system.

    Args:
        omega (float): temporal wavenumber.
        m (int): circumferential wavenumber.
        mach (float): Mach number.
        ri (float): inner radius of a circular annulus.
        ro (float): outer radius of a circular annulus.
        n (int): number of eigenvalues to compute.

    Returns:
        An array of the first 'n' eigenvalues for the system.

    """
    if (n % 2 != 0):
        print('WARNING: number of eigenvalues to find should be divisible by two.')

    # Compute zeros for determinant of solution to convected wave equation
    nzeros = int(n/2)
    zero_loc = compute_zeros(m,mach,ri,ro,nzeros)

    # Compute eigenvalues by solving: zero_loc^2 = (omega + M*k)^2 - k^2  for k
    k = np.zeros(n,dtype=np.complex)
    for i in range(len(zero_loc)):
        a = mach*mach - 1.
        b = 2.*omega*mach
        c = omega*omega - zero_loc[i]*zero_loc[i]
        roots = np.roots([a,b,c])
        k[2*i]   = roots[0]
        k[2*i+1] = roots[1]

    return k



def compute_eigenvector(r,sigma,m,eigenvalue):
    """ Return the eigenvector for the system.

    Args:
        r (np.array(float)): array of radial locations.
        sigma (float): ratio of inner to outer radius, ri/ro.
        m (int): circumferential wavenumber.
        eigenvalue (float): eigenvalue of the eigenvector to be comptued.

    Returns:
        the eigenvector associated with the input 'm' and 'eigenvalue', evaluated at radial locations 
        defined by the input array 'r'. Length of the return array is len(r).

    """
    Q_mn = -sp.jvp(m,sigma*eigenvalue)/sp.yvp(m,sigma*eigenvalue)

    eigenvector = np.zeros(len(r))
    for irad in range(len(r)):
        eigenvector[irad] = sp.jv(m,eigenvalue*r[irad]) + Q_mn*sp.yv(m,eigenvalue*r[irad])

    # Normalize the eigenmode. Find abs of maximum value.
    ev_mag = np.absolute(eigenvector)
    ev_max = np.max(ev_mag)

    eigenvector = eigenvector/ev_max

    return eigenvector











