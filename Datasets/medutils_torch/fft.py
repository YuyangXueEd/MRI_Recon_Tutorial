import numpy as np
import torch

def fft2(img, axes=(-2, -1)):
    """ Compute scaled fft2
    :param img: input image (np.array)
    :param axes: tuple of axes over which the fft2 is computed
    :return: centered and scaled fft2
    """
    assert len(axes) == 2
    axes = list(axes)
    full_axes = list(range(0, img.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    return np.fft.fft2(img, axes=axes) / np.sqrt(img.shape[axes[0]] * img.shape[axes[1]])


def fft2c(img, axes=(-2, -1)):
    """
    Compute centered and scaled fft2
    :param img: input image (np.array)
    :param axes: tuple of axes over which the fft2 is computed
    :return: centered and scaled fft2
    """

    assert len(axes) == 2
    axes = list(axes)
    full_axes = list(range(0, img.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=axes), axes=axes), axes=axes) / np.sqrt(
        img.shape[axes[0]] * img.shape[axes[1]])


def ifft2(img, axes=(-2, -1)):
    """ Compute scaled ifft2
    :param img: input image (np.array)
    :param axes: tuple of axes over which the ifft2 is computed
    :return: centered and scaled ifft2
    """
    assert len(axes) == 2
    axes = list(axes)
    full_axes = list(range(0, img.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    return np.fft.ifft2(img, axes=axes) * np.sqrt(img.shape[axes[0]] * img.shape[axes[1]])


def ifft2c(img, axes=(-2, -1)):
    """
    Compute centered and scaled ifft2
    :param img: input image (np.array)
    :param axes: tuple of axes over which the ifft2 is computed
    :return: centered and scaled ifft2
    """
    assert len(axes) == 2
    axes = list(axes)
    full_axes = list(range(0, img.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img, axes=axes), axes=axes), axes=axes) * np.sqrt(
        img.shape[axes[0]] * img.shape[axes[1]])
