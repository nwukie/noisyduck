import numpy as np
import scipy.special as sp
import scipy.optimize as op




def eigensystem(b,m,Ri,Ro):
    """ Computes the function associated with the eigensystem of the 
    convected wave equation. The location of the zeros for this function 
    correspond to the eigenvalues for the convected wave equation.

        f = J_m(b*Ri)*Y_m(b*Ro)  -  J_m(b*Ro)*Y_m(b*Ri) 

    Args:
        b: radius coordinate
        m: circumferential wave number
        Ri: inner radius of a circular annulus
        Ro: outer radius of a circular annulus

    """
    #if (b > 200.):
    #    print "WARNING: function 'b_eig_fcn' out of range"
    f = sp.jvp(m,b*Ri)*sp.yvp(m,b*Ro) - sp.jvp(m,b*Ro)*sp.yvp(m,b*Ri)
    return f



def compute_eigenvalues(m,n,Ri,Ro):
    """ This procedure compute the analytical eigenvalues for the convected
    wave equation. A uniform set of nodes is created to search for sign changes
    in the value for the eigensystem. When a sign change is detected, it is known
    that a zero is close. The eigensystem is then passed to a bisection routine
    to find the location of the zero. The location corresponds to the eigenvalue
    for the system.

    Args:
        m: circumferential wavenumber.
        n: number of eigenvalues to compute.
        Ri: inner radius of a circular annulus.
        Ro: outer radius of a circular annulus.

    Returns:
        An array of the first 'n' eigenvalues for the system.

    """
    eig = np.zeros(n)
    for rmode in range(n):
        if rmode == 0:
            beta = np.linspace(0.5,200,2000)
        else:
            beta = np.linspace(eig[rmode-1]+0.1,200,20000)

        for i in range(len(beta)):
            f = eigensystem(beta[i],m,Ri,Ro)
            fnew = f
            bnew = beta[i]
            if i > 0 :
                #Test for sign change
                if np.sign(fnew) != np.sign(fold):
                    #Sign changed, so start bisection
                    eig[rmode] = op.bisect(eigensystem,bold,bnew,(m,Ri,Ro))
                    break
                else:
                    fold = fnew
                    bold = bnew
            else:
                fold = fnew
                bold = bnew

    return eig



def compute_eigenvector(r,sigma,m,eigenvalue):
    """ Return the eigenvector for the system.

    Args:
        r: array of radial locations
        sigma: ratio of inner to outer radius, ri/ro
        m: circumferential wavenumber
        eigenvalue: eigenvalue of the eigenvector to be comptued

    Returns:
        the eigenvector associated with the input 'm' and 'eigenvalue', evaluated at radial locations 
        defined by the array 'r'. Length of the return array is len(r).

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











