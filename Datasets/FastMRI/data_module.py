from argparser import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import fastmri
import torch
from fastmri.data import CombinedSliceDataset, SliceDataset

class FastMRIDataModule:
    """
    This class handles configurations for training on fastMRI data.

    It is setup to process configurations independently of training modules.

    Subsampling mask and transform configurations are expected to be done
    by the main client training scripts and passed into this data module.

    For training with `ddp` be sure to set `distributed_sampler=True` to make
    sure that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(self,
                 data_path: Path,
                 challenge: str,
                 train_transform: Callable,
                 val_transform: Callable,
                 test_transform: Callable,
                 combine_train_val: bool = False,
                 test_split: str = "test",
                 test_path: Optional[Path] = None,
                 sample_rate: Optional[float] = None,
                 val_sample_rate: Optional[float] = None,
                 test_sample_rate: Optional[float] = None,
                 volume_sample_rate: Optional[float] = None,
                 val_volume_sample_rate: Optional[float] = None,
                 test_volume_sample_rate: Optional[float] = None,
                 use_dataset_cache_file: bool = True,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 distributed_sampler: bool = Fasle):

        """

        :param data_path: Path to root data directory.
        :param challenge: choices('multicoil', singlecoil')
        :param train_transform: A transform object for the training split
        :param val_transform: A transform object for the validation split
        :param test_transform:
        :param combine_train_val:
        :param test_split:
        :param test_path:
        :param sample_rate:
        :param val_sample_rate:
        :param test_sample_rate:
        :param volume_sample_rate:
        :param val_volume_sample_rate:
        :param test_volume_sample_rate:
        :param use_dataset_cache_file:
        :param batch_size:
        :param num_workers:
        :param distributed_sampler:
        """
        super().__init__()