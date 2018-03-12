# -*- coding: utf-8 -*-
r""" 
Analytical
----------

This module provides a functionality for computing the eigenvalue/eigenvector
decomposition of a uniform axial flow in an annular cylindrical duct. The decomposition
is based on an analytical solution of the convected wave equation for pressure, 
yielding the acoustic part of the eigen decomposition. The eigenvectors from this 
decomposition correspond specifically to the acoustic pressure disturbance.


Theory:
~~~~~~~

The analysis here follows that of Moinier and Giles, "Eigenmode Analysis for 
Turbomachinery Applications", Journal of Propulsion and Power, Vol. 21, No. 6, 2005.

For a uniform axial mean, linear pressure perturbations satisfy the convected wave equation

.. math::

    \left( \frac{\partial}{\partial t} + M \frac{\partial}{\partial z} \right)^2 p = 
    \nabla^2 p \quad\quad\quad \lambda < r < 1


where :math:`M` is the Mach number of the axial mean flow and :math:`r` has been normalized
by the outer duct radius. Boundary conditions for a hard-walled duct imply zero radial 
velocity, which is imposed using the condition


.. math::

    \frac{\partial p}{\partial r} = 0   \quad\quad \text{at}    \quad\quad r=\lambda,1


Conducting a normal mode analysis of the governing equation involves assuming a solution
with a form 

.. math::

    p(r,\theta,z,t) = e^{j(\omega t + m \theta + k z)}P(r)

This leads to a Bessel equation

.. math::

    \frac{1}{r} \frac{d}{dr} \left(r \frac{dP}{dr} \right) + \left( \mu^2 - \frac{m^2}{r^2} 
    \right)P = 0    \quad\quad \lambda < r < 1

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
    
    eigenvalues, eigenvectors = noisyduck.annulus.analytical.decomposition(omega,m,mach,r,n)

"""
import numpy as np
import scipy.special as sp
import scipy.optimize as op



def decomposition(omega,m,mach,r,n):
    """ This procedure computes the analytical eigen-decomposition of 
    the convected wave equation. The eigenvectors returned correspond specifically
    with acoustic pressure perturbations.

    Inner and outer radii are computed using min(r) and max(r), so it is important 
    that these end-points are included in the incoming array of radial locations.

    Args:
        omega (float): temporal wave number.
        m (int): circumferential wave number.
        mach (float): Mach number.
        r (float): array of radius stations, including rmin and rmax.
        n (int): number of eigenvalues/eigenvectors to compute.

    Returns:
        (eigenvalues, eigenvectors): a tuple containing an array of eigenvalues, and an array of eigenvectors evaluated at radial locations.

    """

    if (n % 2 != 0):
        print("WARNING: number of eigenvalues to find should be divisible by two. "
              "Because, for each zero found, there are two associated eigenvalues.")

    ri = np.min(r)
    ro = np.max(r)

    # Compute zeros for determinant of solution to convected wave equation
    # Then use zeros to compute the axial wavenumbers(eigenvalues)
    nzeros = int(n/2)
    zeros = compute_zeros(m,mach,ri,ro,nzeros)
    eigenvalues = compute_eigenvalues(omega,mach,zeros)

    #r = np.linspace(ri,ro,100)
    eigenvectors = np.zeros((len(r),n))
    for i in range(len(zeros)):
        eigenvectors[:,2*i  ] = compute_eigenvector(r,(ri/ro),m,zeros[i])
        eigenvectors[:,2*i+1] = compute_eigenvector(r,(ri/ro),m,zeros[i])

    return eigenvalues, eigenvectors




def eigensystem(b,m,ri,ro):
    r""" Computes the function associated with the eigensystem of the 
    convected wave equation. The location of the zeros for this function 
    correspond to the eigenvalues for the convected wave equation.

    The solution to the Bessel equation with boundary conditions applied
    yields a system of two linear equations.
    .. math::

        A x = 
        \begin{bmatrix}
            J'_m(\mu \lambda)   &   Y'_m(\mu \lambda) \\
            J'_m(\mu)           &   Y'_m(\mu)
        \end{bmatrix}
        \begin{bmatrix}
            x_1 \\ x_2
        \end{bmatrix}
        = 0

    This procedure evaluates the function
    .. math::

        det(A) = f(b) = J_m(b*ri)*Y_m(b*ro)  -  J_m(b*ro)*Y_m(b*ri) 

    So, this procedure can be passed to another routine such as numpy
    to find zeros.

    Args:
        b (float): coordinate.
        m (int): circumferential wave number.
        ri (float): inner radius of a circular annulus.
        ro (float): outer radius of a circular annulus.

    """
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
            #beta = np.linspace(0.5,200,2000)
            beta = np.linspace(0.000001,100,2000)
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





def compute_eigenvalues(omega,mach,zeros):
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
    # Compute eigenvalues by solving: zeros^2 = (omega + M*k)^2 - k^2  for k
    k = np.zeros(2*len(zeros),dtype=np.complex)
    for i in range(len(zeros)):
        a = mach*mach - 1.
        b = 2.*omega*mach
        c = omega*omega - zeros[i]*zeros[i]
        roots = np.roots([a,b,c])
        k[2*i]   = roots[0]
        k[2*i+1] = roots[1]

    return k



def compute_eigenvector(r,sigma,m,zero):
    """ Return the eigenvector for the system.

    Args:
        r (np.array(float)): array of radial locations.
        sigma (float): ratio of inner to outer radius, ri/ro.
        m (int): circumferential wavenumber.
        zero (float): a zero of the determinant of the convected wave equation

    Returns:
        the eigenvector associated with the input 'm' and 'zero', evaluated at radial locations 
        defined by the input array 'r'. Length of the return array is len(r).

    """
    Q_mn = -sp.jvp(m,sigma*zero)/sp.yvp(m,sigma*zero)

    eigenvector = np.zeros(len(r))
    for irad in range(len(r)):
        eigenvector[irad] = sp.jv(m,zero*r[irad]) + Q_mn*sp.yv(m,zero*r[irad])

    # Normalize the eigenmode. Find abs of maximum value.
    ev_mag = np.absolute(eigenvector)
    ev_max = np.max(ev_mag)

    eigenvector = eigenvector/ev_max

    return eigenvector











