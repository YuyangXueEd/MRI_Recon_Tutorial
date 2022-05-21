import pathlib
import random

import h5py
import numpy as np
import torch.utils.data import Dataset

class SliceData(Dataset):
    """
    A Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes
             the raw data into appropriate form. The transform function
             should take 'kspace', 'target', 'attributes', 'filename',
             and 'slice' as inputs. 'target' may be null for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """

        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        self.examples = []
        files = list(pathlib.Path(root).iterdir())

        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor.
    """
    return torch.view_as_real(torch.from_numpy(data))


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have
             at least 3 dimensions, where dimensions -3 and -2 are the
             spatial dimensions, and the final dimension has size 2
             (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints)
             and a random number seed and returns a mask.
        seed (int or 1d array like, optional): seed for the random generator.

    Returns:
        (tuple):
        masked_data (torch.Tensor): Subsampled k-space data
        mask (torch.Tensor): The generated mask
    """

    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return torch.where(mask==0, torch.Tensor([0], data), mask)


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = torch.fft.ifftshift(data, dim=(-3, -2))
    data = torch.fft.ifft(data, 2, normalized=True)
    data = torch.fft.fftshift(data, dim=(-3, -2))
    return data


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = torch.fft.ifftshift(data, dim=(-3, -2))
    data = torch.fft.fft(data, 2, normalized=True)
    data = torch.fft.fftshift(data, dim=(-3, -2))
    return data


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images
    Args:
        data (torch.Tensor): The complex input tensor to be center cropped.
              It should have at least 3 dimensions and the cropping is applied
              along dimensions -3 and -2 and the last dimensions should have a
              size of 2.
        shape (int, int): The output shape. The shape should be smaller than
              the correpsonding dimensions of data.
    """

    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def complex_abs(data):
    """
    Compute the absolute value of complex data

    Args:
        data (torch.Tensor): A complex valued tensor, where
             the size of the final dimension should be 2.

    Returns:
        torch.Tensor: Absolute value
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std

