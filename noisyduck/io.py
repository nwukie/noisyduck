import numpy as np


def write_decomposition(eigenvalues, eigenvectors_l, eigenvectors_r, r, filename):
    """ Write the result of a numerical eigen-decomposition to file.

    Args:
        eigenvalues (complex): array of eigenvalues of the decomposition.
        eigenvectors (np.array([float,float])): rank-2 array of eigenvectors from the decomposition. [len(r),len(nvectors)].
        r (float): array of radial coordinates from the discretization that the eigenvectors were evaluated at.

    """
    # Open new file
    handle = open(filename,'w')

    handle.write('&sizes\n')
    handle.write('  nr       = '+str(len(r))+'\n')
    handle.write('  nvectors = '+str(len(eigenvalues))+'\n')
    handle.write('/\n')
    
    handle.write('&eigendecomposition\n')
    handle.write('  k = ')
    for i in range(len(eigenvalues)):
        handle.write('('+str(eigenvalues[i].real)+','+str(eigenvalues[i].imag)+')')
        if i == (len(eigenvalues)-1):
            handle.write('\n')
        else:
            handle.write(', ')
#    handle.write('/\n')


    handle.write('  r = ')
    for i in range(len(r)):
        handle.write(str(r[i]))
        if i == (len(r)-1):
            handle.write('  \n')
        else:
            handle.write(', ')


    # write right eigenvectors
    for vec in range(eigenvectors_r.shape[1]):
        handle.write('  A(1:'+str(eigenvectors_r.shape[0])+','+str(vec+1)+') = ')
        for val in range(eigenvectors_r.shape[0]):
            handle.write('('+str(eigenvectors_r[val,vec].real)+','+str(eigenvectors_r[val,vec].imag)+')')
            if val == eigenvectors_r.shape[0]-1:
                handle.write('\n')
            else:
                handle.write(', ')

    # write left eigenvectors
    for vec in range(eigenvectors_l.shape[1]):
        handle.write('  B(1:'+str(eigenvectors_l.shape[0])+','+str(vec+1)+') = ')
        for val in range(eigenvectors_l.shape[0]):
            handle.write('('+str(eigenvectors_l[val,vec].real)+','+str(eigenvectors_l[val,vec].imag)+')')
            if val == eigenvectors_l.shape[0]-1:
                handle.write('\n')
            else:
                handle.write(', ')
    handle.write('/\n')


    handle.close()


