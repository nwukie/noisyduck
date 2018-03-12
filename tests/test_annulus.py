#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing decomposition results for cylindrical annulus.

    Fixtures:
        uniform_state:      returns dictionary containing the state for a uniform axial 
                            mean flow.
        uniform_numerical:  returns eigenvalues and eigenvectors from the numerical 
                            decomposition of the uniform axial mean flow.
        uniform_analytical: returns eigenvalues and eigenvectors from the analytical
                            decomposition of the uniform axial mean flow.

    The fixtures here essentially precompute results, then these results are
    used in tests for various purposes.

"""
from __future__ import division
import pytest
import numpy as np
import noisyduck as nd




@pytest.fixture
def uniform_state():
    """Return the mean state for uniform axial flow."""
    ri=0.25
    ro=1.0
    res=50
    r=np.linspace(ri,ro,res)
    return {'gam':1.4, 'rho':1.2, 'vr' :0., 'vt' :0., 'vz' :100., 'p':100000., 'omega':3000., 'm':2, 'r':r}




@pytest.fixture
def uniform_numerical(uniform_state):
    """Compute the numerical eigendecomposition of a uniform axial 
    mean flow that can be used by tests.
    """
    
    evals, evecs_l, evecs_r = nd.annulus.numerical.decomposition(uniform_state['omega'],
                                                                 uniform_state['m'],
                                                                 uniform_state['r'],
                                                                 uniform_state['rho'],
                                                                 uniform_state['vr'],
                                                                 uniform_state['vt'],
                                                                 uniform_state['vz'],
                                                                 uniform_state['p'],
                                                                 uniform_state['gam'],
                                                                 filter='acoustic',alpha=0.0000001,perturb_omega=False)
    return evals, evecs_l, evecs_r




@pytest.fixture
def uniform_analytical(uniform_state,uniform_numerical):
    """ Compute the analytical eigendecomposition of a uniform axial
    mean flow that can be used by tests.
    """
    gam  = uniform_state['gam']
    p    = uniform_state['p']
    rho  = uniform_state['rho']
    c    = np.sqrt(gam*p/rho)
    mach = uniform_state['vz']/c

    # Access the results of the numerical decomposition so we know how many
    # eigenvalues to find.
    evals_n, evecs_ln, evecs_rn = uniform_numerical
    n = len(evals_n)

    evals, evecs_r = nd.annulus.analytical.decomposition(uniform_state['omega']/c,
                                                         uniform_state['m'],
                                                         mach,
                                                         uniform_state['r'],
                                                         n)

    return evals, evecs_r





def test_uniformaxialflow_analytical_unique(uniform_analytical):
    """ Test the eigenvalues from the analytical decomposition are unique """
    evals, evecs_r = uniform_analytical
    assert len(evals) == len(set(evals))




def test_uniformaxialflow_analytical_matches_reference(uniform_analytical):
    """ Test the analytical eigenvalues match previously computed reference values. """

    evals_a, evecs_ra = uniform_analytical

    # Previously computed reference eigenvalues for the analytical
    # eigendecomposition of a uniform axial mean flow in an annular
    # duct. rho=1.2, vz=100, p=100000, omega=3000/c, m=2, ri/ro=0.25
    evals_ref = np.array([(11.888859931191348+0j),
                          (-6.263859931191352+0j),
                          (9.746593068700136+0j),
                          (-4.121593068700137+0j),
                          (2.8124999999999987+3.0004194877589363j),
                          (2.8124999999999987-3.0004194877589363j),
                          (2.812499999999999+10.162007091553026j),
                          (2.812499999999999-10.162007091553026j),
                          (2.8124999999999982+15.38697739666744j),
                          (2.8124999999999982-15.38697739666744j)])


    eval_matches = np.zeros(len(evals_a), dtype=bool)
    for i in range(len(evals_a)):
        for j in range(len(evals_ref)):
            if np.isclose(evals_a[i],evals_ref[j], atol=1.e-6):
                eval_matches[i]=True
                break

    # Assert that we found a matching reference eigenvalue for each
    # analytical eigenvalue.
    assert eval_matches.all()


def test_uniformaxialflow_numerical_matches_analytical(uniform_analytical,uniform_numerical):
    """ Test the annular cylindrical duct decomposition for the
    case of uniform axial mean flow.
    """
    # Unpack results from analytical and numerical decompositions
    evals_a, evecs_ra           = uniform_analytical
    evals_n, evecs_ln, evecs_rn = uniform_numerical

    # For each numerical eigenvalue, test that it matches closely
    # with one of the analytical eigenvalues.
    eval_matches = np.zeros(len(evals_n), dtype=bool)
    for i in range(len(evals_n)):
        # Search each entry in evals_a to try and find a close match
        for j in range(len(evals_a)):
            if np.isclose(evals_n[i],evals_a[j],atol=1.e-2): 
                eval_matches[i]=True 
                break

    # Assert that we found a matching analytical eigenvalue for each 
    # numerical eigenvalue
    assert eval_matches.all()


