import numpy as np


# 15-point Dispersion Relation Preserving Filter
def drp15(f):
    """ 15-point DRP filter from "Computational Aeroacoustics", Tam. Strongly
    filters low-wavenumber components of a signal. End points are handled by
    creating an artificial reflection of the origin signal about the end points.
    In this way, the central filter has 'ghost' information past the end points
    that it needs to construct the filter.

    Args:
        f: incoming function

    Returns:
        f_tilde: filtered function
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



