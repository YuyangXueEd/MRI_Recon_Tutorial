import logging
import pathlib
import random
import shutil
import time

from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fastmri.data.subsample import MaskFunc
from fastmri.data.transforms import to_tensor
from fastmri.data.mri_data import SliceData
from Models import unet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Args(ArgumentParser):

    def __init__(self, **override):
        super(Args, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument("--mode", default="train", choices=("train", "test"), type=str, help="Operation mode")
        # dataset
        self.add_argument("--data-split", choices=['val', 'test_v2'], required=True,
                          help="which data partition to run on")
        self.add_argument("--seed", default=42, type=int, help='seed for random number generation.')
        self.add_argument("--resolution", default=320, type=int, help="resolution of images")
        self.add_argument("--challenge", default="singlecoil", choices=["singlecoil", "multicoil"], required=True,
                          help="which challenge.")
        self.add_argument("--data_path", type=pathlib.Path, required=True, help='Path to the dataset')
        self.add_argument("--sample_rate", type=float, default=1., help='Fraction of total volumes to include')
        # mask
        self.add_argument("--center_fractions", default=[0.08, 0.04], nargs="+", type=float,
                          help="Number of center lines to use in mask")
        self.add_argument("--accelerations", default=[4, 8], nargs="+", type=int,
                          help="Acceleration rates to use for masks")
        self.add_argument("--mask-kspace", action='store_true',
                          help="Whether to apply a mask (set to True for val data and False for test data")
        # model
        self.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to the U-net model")
        self.add_argument("--out_dir", type=pathlib.Path, required=True, help="Path to save the reconstructions to")
        self.add_argument("--batch_size", default=32, type=int, help="Mini-batch size")
        self.add_argument("--device", default='cuda', type=str, help='Which device to run on')

        self.set_defaults(**override)

class DataTransform:
    def __init__(self, mask_func, resolution, challenge, use_seed=True):
        """

        :param mask_func: fastmri.data.subsample.MaskFunc
        :param resolution: resolution of the image
        :param challenge: "singlecoil" or "multicoil"
        :param use_seed: a pseudo random number generator seed from the filename.
                         This ensures that the same mask is used for a given volume every time.
        """

        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.resolution =resolution
        self.challenge = challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """

        :param kspace (numpy.array): input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                                     or (rows, cols, 2) for single coils data.
        :param target (numpy.array): Target image
        :param attrs (dict): Acquisition related information stored in the h5 obj
        :param fname (str): file name
        :param slice (int): serial number of the slice
        :return (tuple):
               image (torch.Tensor): zero-filled input image
               target (torch.Tensor): Target image converted to a torch tensor.
               mean (float): mean value used for normalisation.
               std (float): std used for normalisation.
               norm (float): l2 norm of the entire volume.
        """

        kspace = to_tensor(kspace)
        # Apply mask
        seed = None if not self.use_seed