import numpy as np
import torch
from torch import nn


def complex_abs(data, dim=-1, keepdim=False, eps=0):
    assert data.size(dim == 2)
    return (data ** 2 + eps).sum(dim=dim, keepdim=keepdim).sqrt()


def complex_mult(data1, data2, dim=-1):
    """
    Element-wise complex matrix multiplication X^T Y

    :param data1 -> torch.Tensor:
    :param data2 -> torch.Tensor:
    :param dim -> int: dimension that represents the complex values
    """

    assert data1.size(dim) == 2
    assert data2.size(dim) == 2
    re1, im1 = torch.unbind(data1, dim=dim)
    re2, im2 = torch.unbind(data2, dim=dim)

    return torch.stack([re1 * re2 - im1 * im2, im1 *re2 + re1 * im2], dim=dim)


def complex_mult_conj(data1, data2, dim=-1):
    """
    Element-wise complex matrix multiplication with conjugation x^H Y

    :param data1 -> torch.Tensor:
    :param data2 -> torch.Tensor:
    :param dim -> int: dimension that represents the complex values
    """
    assert data1.size(dim) == 2
    assert data2.size(dim) == 2
    re1, im1 = torch.unbind(data1, dim=dim)
    re2, im2 = torch.unbind(data2, dim=dim)

    return torch.stack([re1 * re2 - im1 * im2, im1 * re2 - re1 * im2], dim=dim)


def complex_div(data1, data2, dim=-1):
    """
    Element-wise division x^H Y

    :param data1 -> torch.Tensor:
    :param data2 -> torch.Tensor:
    :param dim -> int: dimension that represents the complex values
    """
    assert data1.size(dim) == 2
    assert data2.size(dim) == 2
    re1, im1 = torch.unbind(data1, dim=dim)
    re2, im2 = torch.unbind(data2, dim=dim)

    return  torch.stack([re1 * re2 + im1 * im2, im1 * re2 - re1 * im2], dim=dim) / complex_abs(data2, keepdim=True) ** 2


def complex_dotp(data1, data2):
    """
        Complex dot product

        :param data1 -> torch.Tensor:
        :param data2 -> torch.Tensor:
        :param dim -> int: dimension that represents the complex values
        """

    assert data1.size(-1) == 2
    assert data2.size(-1) == 2

    mult = complex_mult_conj(data1, data2)
    re, im = torch.unbind(mult, dim=-1)
    return torch.stack([torch.sum(re), torch.sum(im)])


def complex_pseudocovariance(data):
    """
    :param data: Data variable has to be already mean-free!
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
    cxx = np.sum(re * re) / (N - 1)
    cyy = np.sum(im * im) / (N - 1)
    cxy = np.sum(re * im) / (N - 1)

    # Eigenvalue decomposition
    # C = V * S * inv(V)
    s1 = (cxx + cyy) / 2 - np.sqrt((cxx + cyy) ** 2 / 4 - cxx * cyy + cxy ** 2)
    s2 = (cxx + cyy) / 2 + np.sqrt((cxx + cyy) ** 2 / 4 - cxx * cyy + cxy ** 2)

    # compute eigenvectors
    v1x = s1 - cyy
    v1y = cxy
    v2x = s2 - cyy
    v2y = cxy

    # normalise eigenvectors
    norm1 = np.sqrt(np.sum(v1x * v1x + v1y * v1y))
    norm2 = np.sqrt(np.sum(v2x * v2x + v2y * v2y))

    v1x = v1x.copy() / norm1
    v1y = v1y.copy() / norm1

    v2x = v2x.copy() / norm2
    v2y = v2y.copy() / norm2

    # now we need the sqrt of the covariance matrix.
    # C^{-0.5} = V * sqrt(S) * inv(V)
    det = v1x * v2y - v2x * v1y
    s1 = np.sqrt(s1) / det
    s2 = np.sqrt(s2) / det

    cov_xx_half = v1x * v2y * s1 - v1y * v2x * s2
    cov_yy_half = v1x * v2y * s2 - v1y * v2x * s1
    cov_xy_half = v1x * v2x * (s2 - s1)
    cov_yx_half = v1y * v2y * (s1 - s2)

    return cov_xx_half, cov_xy_half, cov_yx_half, cov_yy_half


def matrix_invert(xx, xy, yx, yy):
    det = xx * yy - xy * yx
    return yy / det, -xy / det, -yx / det, xx / det


def normalise(x, mean, conv_xx_half, cov_xy_half, cov_yx_half, cov_yy_half):
    x_m = x - mean
    re = np.real(x_m)
    im = np.imag(x_m)

    cov_xx_half_inv, cov_xy_half_inv, cov_yx_half_inv, cov_yy_half_inv = matrix_invert(conv_xx_half, cov_xy_half,
                                                                                       cov_yx_half, cov_yy_half)
    x_norm_re = cov_xx_half_inv * re + cov_xy_half_inv * im
    x_norm_im = cov_yx_half_inv * re + cov_xy_half_inv * im
    img = x_norm_re + 1j * x_norm_im

    return img
