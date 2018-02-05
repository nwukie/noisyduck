import numpy as np

# 15-point Dispersion Relation Preserving Filter
def drp15(f):
    """ 15-point DRP filter from "Computational Aeroacoustics", Tam. Strongly
    filters low-wavenumber components of a signal. End points are handled by
    creating an artificial reflection of the origin signal about the end points.
    In this way, the central filter has 'ghost' information past the end points
    that it needs to construct the filter.

    Args:
        f (np.array(float)): discrete values of f evaluated on a uniform grid.

    Returns:
        f_tilde (np.array(float)): filtered function
    """

    # 15-point DRP filter coefficients from "Computational Aeroacoustics", Tam
    d0  =  0.2042241813072920
    dp1 = -0.1799016298200503
    dm1 = -0.1799016298200503
    dp2 =  0.1224349282118140
    dm2 =  0.1224349282118140
    dp3 = -6.3456279827554890e-2
    dm3 = -6.3456279827554890e-2
    dp4 =  2.4341225689340974e-2
    dm4 =  2.4341225689340974e-2
    dp5 = -6.5519987489327603e-3
    dm5 = -6.5519987489327603e-3
    dp6 =  1.1117554451990776e-3
    dm6 =  1.1117554451990776e-3
    dp7 = -9.0091603462069583e-5
    dm7 = -9.0091603462069583e-5
    coeffs = [dm7, dm6, dm5, dm4, dm3, dm2, dm1, d0, dp1, dp2, dp3, dp4, dp5, dp6, dp7]

    # Create storage for result
    f_tilde = np.copy(f)

    # Duplicate waveform to handle end points
    two = np.append(np.flipud(f),f)
    three = np.append(two,np.flipud(f))

    # Apply filter
    for j in range(len(f)):
        dat = three[(len(f)+j-7):(len(f)+1+j+7)]
        f_tilde[j] = np.sum(dat*coeffs)

    return f_tilde




def physical(eigenvalues,eigenvectors,r,alpha_cutoff=0.00001,filters='acoustic'):
    """ Procedure for filtering/sorting eigenvectors into physical categories.
    Generally, we are trying to determine if a given eigenvector is a convected
    wave, an upstream/downstream traveling acoustic wave, or a spurious mode.

    Args:
        eigenvalues (complex): Array of eigenvalues.
        eigenvectors (complex): Array of eigenvectors corresponding to the eigenvalues. [len(r)*nfields,len(eigenvalues)].
        r (float): Array of radial coordinate locations where the eigenvectors have been evaluated at.
        alpha_cutoff (float, optional): A cutoff criteria for filtering. Higher alpha will include higher wavenumber modes.
        filters (string, optional): Select how to filter the incoming eigenvalue/eigenvector pairs.

    Returns:
        (eigenvalues, eigenvectors): a tuple containing an array of filtered eigenvalues, and their corresponding eigenvectors.
    """
    # Separate eigenvectors into primitive variables
    res = len(r)
    rho_eigenvectors = eigenvectors[0*res:1*res,:]
    u_eigenvectors   = eigenvectors[1*res:2*res,:]
    v_eigenvectors   = eigenvectors[2*res:3*res,:]
    w_eigenvectors   = eigenvectors[3*res:4*res,:]
    p_eigenvectors   = eigenvectors[4*res:5*res,:]

    if (filters == 'acoustic'):
        # Work with pressure component of eigenvectors
        # Create separate copy so we can compare before/after modes
        pr_eigenvectors   = np.copy(np.real(p_eigenvectors))
        pr_eigenvectors_f = np.copy(pr_eigenvectors)

        # Call filter for each eigenvector
        for i in range(pr_eigenvectors.shape[1]):
            pr_eigenvectors_f[:,i] = drp15(pr_eigenvectors[:,i])

        # Select low-wavenumber modes based on the ratio of their vector norms
        amp   = np.zeros(pr_eigenvectors.shape[1])
        amp_f = np.zeros(pr_eigenvectors.shape[1])
        for i in range(pr_eigenvectors.shape[1]):
            for j in range(len(r)-1):
                amp[i]   = amp[i]   + 0.5*(r[j+1]-r[j])*(r[j]*pr_eigenvectors[  j,i]*pr_eigenvectors[  j,i]  +  r[j+1]*pr_eigenvectors[  j+1,i]*pr_eigenvectors[  j+1,i])
                amp_f[i] = amp_f[i] + 0.5*(r[j+1]-r[j])*(r[j]*pr_eigenvectors_f[j,i]*pr_eigenvectors_f[j,i]  +  r[j+1]*pr_eigenvectors_f[j+1,i]*pr_eigenvectors_f[j+1,i])

        # Evaluate wavenumber selection criteria: alpha
        alpha = np.zeros(len(amp))
        for i in range(len(amp)):
            if (amp[i] < np.finfo(float).eps):
                # Handle case where initial amplitude for pressure was zero. 
                # We aren't interested in those modes for acoustics so we set their metric to 
                # very large, effectively filtering them out.
                alpha[i] = np.finfo(float).max
            else:
                alpha[i] = amp_f[i]/amp[i]

        # Evaluate pressure magnitude selection criteria
        # Find 2*res eigenvectors with largest |p'|
        # np.argpartition just returns the indices
        for i in range(len(amp)):
            largest = np.argpartition(amp, -2*res)[-2*res:]


        # Flag eigenvectors that satisfy criteria: 
        #   1: Wavenumber filter criteria: alpha < alpha_cutoff.
        #   2: Pressure magnitude criteria, only 2*res largest are kept.
        keep = []
        for i in range(len(alpha)):
            if (np.abs(alpha[i]) < alpha_cutoff) and (i in largest):
                # Store as structured array so we can sort the entire system by alpha
                tmp = np.array([(i,alpha[i])],dtype=[('index',np.int),('alpha',np.float)])
                if (len(keep) == 0):
                    keep = tmp
                else:
                    keep = np.append(keep,tmp, axis=0)
        
        # Sort eigenmodes and indices being kept by 'alpha'
        keep = np.sort(keep,order='alpha')

        # Collect eigenvalues/eigenvectors flagged to pass the filter
        nmodes = keep.shape[0]
        eigenvalues_f  = np.zeros([nmodes],       dtype=np.complex)
        eigenvectors_f = np.zeros([5*res,nmodes], dtype=np.complex)
        for i in range(nmodes):
            eigenvalues_f[i]    = eigenvalues[keep['index'][i]]
            eigenvectors_f[:,i] = eigenvectors[:,keep['index'][i]]

        return eigenvalues_f, eigenvectors_f
        

