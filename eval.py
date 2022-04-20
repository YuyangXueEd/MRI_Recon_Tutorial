import sys
import pathlib
import argparse
from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from fastmri.data.mri_data import SliceData
from fastmri.data.subsample import MaskFunc
from fastmri.data.transforms import to_tensor, apply_mask, \
    complex_center_crop, complex_abs, \
    normalize_instance
from fastmri.coil_combine import rss
from fastmri.utils import save_reconstrctions
from fastmri import fftc

from Models import Unet


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

    def __init__(self, resolution, challenge, mask_func=None):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.resolution = resolution
        self.challenge = challenge
        self.mask_func = mask_func

    def __call__(self, kspace, target, attrs, fname, slice):
        """

        :param kspace (numpy.Array): k-space measurements
        :param target (numpy.Array): Target image
        :param attrs (dict): Acquisition related information stored in the H5 obj
        :param fname (pathlib.Path): Path to the input file
        :param slice (int): Serial number of the slice
        :return:
        image (torch.Tensor): Normalized zero-filled input image
        mean (float): Mean of the zero-filled image
        std (float): standard deviation of the zero-filled image
        fname (pathlib.Path): Path to the input file
        slice (int): Serial number of the slice
        """

        kspace = transforms.to_tensor(kspace)
        if self.mask_func is not None:
            seed = tuple(map(ord, fname))
            masked_kspace, _ = transforms.apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # Inverse Fourier Transform to get zero filled solution
        image = fftc.fft2c_new(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # get absolute value
        image = transforms.complex_abs(image)
        # Apple RSOS if multicoil data
        if self.challenge == 'multicoil':
            image = rss(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image)
        image = image.clamp(-6, 6)
        return image, mean, std, fname, slice


def create_data_loaders(args):
    mask_func = None
    if args.mask_kspace:
        mask_func = MaskFunc(args.center_fractions, args.acceleartions)

    data = SliceData(
        root=args.data_path / f'{args.challenge}_{args.data_split}',
        transforms=DataTransform(args.resolution, args.challenge, mask_func),
        sample_rate=1.,
        challenge=args.challenge
    )

    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True
    )

    return data_loader


def load_model(ckpt_file, num):
    ckpt = torch.load(ckpt_file)
    args = ckpt['args']
    model = Unet(1, 1, args.feats, args.num_pool_layers, args.drop_prob).to(args.device)
    if args.data_parallel:
        # TODO add card control
        gpu_num = num
        model = torch.nn.DataParallel(model)
    model.load_state_dict(ckpt['model'])

    return model


def eval_model(args, model, data_loader):
    model.eval()
    recons = defaultdict(list)
    with torch.no_grad():
        for (input, mean, std, fnames, slices) in data_loader:
            input = input.unsqueeze(1).to(args.device)
            out = model(input).to('cpu').squeeze(1)
            for i in range(out.shape[0]):
                out[i] = out[i] * std[i] + mean[i]
                recons[fnames[i]].append((slice[i].numpy(),
                                          out[i].numpy()
                                          ))

    recons = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)]) for name, slice_preds in recons.items()

    }

    return recons


if __name__ == '__main__':
    parser = Args().parse_args(sys.argv[1:])
    data_loader = create_data_loaders(parser)
    model = load_model(args.checkpoint)
    save_reconstructions(recons, args.out_dir)
