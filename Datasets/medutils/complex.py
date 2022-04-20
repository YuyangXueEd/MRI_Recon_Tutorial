import numpy as np


def complex_pseudocovariance(data):
    """
    param data: Data variable has to be already mean-free!
                  Operates on image x of size
                  [nBatch, nSmaps, nFE, nPE, 2]

    A complex number can be shown as $z=x+iy$,
    The mean is:
        $$
        \mu_z = E\{z\}=E\{x\}+iE\{y\}=\mu_x+i\mu_y
        $$
    The variance is:
        $$
        R_{zz} =E\left\{( z-\mu _{z})( z-\mu _{z})^{H}\right\} =R_{xx} +R_{yy} +i\left( R_{xy}^{T} -R_{xy}\right)
        $$
    The pseudocovariance is:
        $$
        P_{zz} =E\left\{( z-\mu _{z})( z-\mu _{z})^{T}\right\} =R_{xx} -R_{yy} +i\left( R_{xy}^{T} +R_{xy}\right)
        $$

    return:
        
    """

    # compute number of elements
    N = data.size

    # separate real/imaginary channel
    re = np.real(data)
    im = np.imag(data)

    # compute covariance entries: cxy=cyx
    cxx = np.sum(re * re) / (N-1)
    cyy = np.sum(im * im) / (N-1)
    cxy = np.sum(re * im) / (N-1)
