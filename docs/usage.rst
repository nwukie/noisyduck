=====
Usage
=====

To use Noisy Duck in a project::

    import noisyduck




Example: Annular Cylindrical Duct 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numerical eigenvalue/eigenvector decomposition:

.. code-block:: python

    import noisyduck as nd
    eigenvalues, eigenvectors = nd.annulus.numerical.decomposition(omega,m,r,rho,u,v,w,p,gam,filter='acoustic')



Analytical eigenvalue/eigenvector decomposition:

.. code-block:: python

    import noisyduck as nd
    eigenvalues, eigenvectors = nd.annulus.analytical.decomposition(omega,m,mach,ri,ro,n)

See :download:`cylindrical_annulus_uniform_flow.py <../examples/cylindrical_annulus_uniform_flow.py>`.


-------------------------
Cylindrical Annuluar Duct
-------------------------

.. automodule:: noisyduck.annulus.analytical
    :members:

.. automodule:: noisyduck.annulus.numerical
    :members:



