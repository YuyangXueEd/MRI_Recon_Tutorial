import numpy as np

import logging
import argparse
import os
import shutil
import time
from os.path import join
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision

from Datasets.FastMRI.subsample import MaskFunc
from Datasets.FastMRI.data import mri

from utils.loss import CompoundLoss

from Models import DCCNN


class DataTransform:
    def __init__(self, mask_func, resolution, challenge, use_seed=True):
        """
        Args:
            mask_func (subsample.MaskFunc): A function that can create a mask for appropriate shape.
            resolution (int): Resolution of the image
            challenge (str): singlecoil or multicoil
            use_seed (bool): If true, this class computes a pseudo random number generator
                       seed from the filename. This ensures that the same mask is used for all
                       the slices of a given volume every time.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.challenge = challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coil, rows, cols, 2)
                    for multi-coil data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object
            fname (str): File name
            slice (int): Serial number of the slice
        Returns:
            (tuple):
            image (torch.Tensor): Zero filled input image
            target (torch.Tensor): Target image converted to a torch Tensor
            mean (float): Mean value used for normalization
            std (float): Standard deviation value used for normalization
            norm (float): L2 norm of the entire volume.
        """

        kspace = mri.to_tensor(kspace)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = mri.apply_mask(kspace, self.mask_func, seed)
        # IFT to get zero filled solution
        image = mri.ifft2(masked_kspace)
        image = mri.complex_center_crop(image, (self.resolution, self.resolution))
        image = mri.complex_abs(image)

        if self.challenge == 'multicoil':
            image = mri.root_sum_of_squares(image)

        image, mean, std = mri.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        target = mri.to_tensor(target)
        target = mri.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        return image, target, mean, std, attrs['norm'].astype(np.float32)


def create_dataset(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    val_mask = MaskFunc(args.center_fractions, args.accelerations)

    train_data = mri.SliceData(
        root=args.train_path,
        transform=DataTransform(train_mask, args.resolution, args.challenge),
        sample_rate=args.sample_rate,
        challenge=args.challenge
    )

    val_data = mri.SliceData(
        root=args.val_path,
        transform=DataTransform(val_mask, args.resolution, args.challenge),
        sample_rate=args.sample_rate,
        challenge=args.challenge
    )

    return val_data, train_data


def create_data_loaders(args):
    val_data, train_data = create_dataset(args)
    display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, val_loader, display_loader


def build_model(args):
    # network
    if args.model_name == 'dccnn':
        net = DCCNN(n_iter=8).to(args.device)
    else:
        raise (NotImplementedError("No model " + args.model_name))

    print('Total # of model params: %.5fM' % (sum(p.numel() for p in net.parameters()) / 10. ** 6))

    return net


def load_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = torch.optim.RMSprop(model.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        input, target, mean, std, norm = data
        input = input.unsqueeze(1).to(args.device)
        target = target.to(args.device)

        output = model(input).squeeze(1)
        loss = F.smooth_l1_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.unsqueeze(1).to(args.device)
            target = target.to(args.device)
            output = model(input).squeeze(1)

            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            target = target * std + mean
            output = output * std + mean

            norm = norm.unsqueeze(1).unsqueeze(2).to(args.device)
            loss = F.mse_loss(output / norm, target / norm, size_average=False)
            losses.append(loss.item())
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            output = model(input)
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def solve(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr, weight_decay=args.weight_decay)

        best_dev_loss = 1e9
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
        visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


# class Solver():
#     def __init__(self, args):
#         torch.autograd.set_detect_anomaly(True)
#         self.args = args
#
#         ## experiment settings
#         self.model_name = self.args.model
#         self.acc = self.args.acc
#         self.imageDir_train = self.args.train_path
#         self.imageDir_val = self.args.val_path
#         self.imageDir_test = self.args.test
#         self.num_epoch = self.args.num_epoch
#         self.batch_size = self.args.batch_size
#         self.val_on_epochs = self.args.val_on_epochs
#         self.resume = self.args.resume
#
#         ## optimizer setting
#         self.lr = self.args.lr
#
#         ## preprocessing setting
#         self.img_size = (192, 160)
#         self.saveDir = 'weight'  # model save path while training
#         if not os.path.isdir(self.saveDir):
#             os.makedirs(self.saveDir)
#
#         self.task_name = self.model_name + '_acc_' + str(self.acc) + '_bs_' + str(self.batch_size) + '_lr_' + str(
#             self.lr)
#         print('task name: ', self.task_name)
#         self.model_path = 'weight/' + self.task_name + '_best.pth'  # model load path
#
#         ## network
#         if self.model_name == 'dccnn':
#             self.net = DCCNN(n_iter=8)
#         else:
#             raise (NotImplementedError("No model " + self.model_name))
#
#         print('Total # of model params: %.5fM' % (sum(p.numel() for p in self.net.parameters()) / 10. ** 6))
#
#         self.net.cuda()
#
#     def train(self):
#         ## Losses
#         self.criterion = CompoundLoss('ms-ssim')
#         self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-3, weight_decay=1e-10)
#
#         # load data
#         dataset_train, dataset_val, dataset_display = create_data_loaders(args)
#
#         self.writer = SummaryWriter('log/' + self.task_name)
#
#         ## run epochs
#         start_epoch = 0
#         best_val_psnr = 0
#         if self.resume:
#             best_name = self.task_name + '_best.pth'
#             checkpoint = torch.load(join(self.saveDir, best_name))
#             self.net.load_state_dict(checkpoint['net'])
#             start_epoch = checkpoint['epoch'] + 1
#             best_val_psnr = checkpoint['val_psnr']
#             print('load pretrained model---, start epoch at, ', start_epoch, ', star_psnr_val is: ', best_val_psnr)
#
#         for epoch in range(start_epoch, self.num_epoch):
#             self.net.train()
#             for data_dict in tqdm(dataset_train):


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Experiement settings
    parser.add_argument('--mode',
                        default='train',
                        choices=['train', 'test'],
                        help='mode for the program')
    parser.add_argument('--model',
                        default='dccnn',
                        choices=['dccnn', 'pldnet', 'hqsnet'],
                        help='models to reconstruct')
    parser.add_argument('--acc',
                        type=int,
                        default=4,
                        help='Acceleration factor for k-space sampling')
    ## Dataset
    parser.add_argument('--train_path',
                        default='data/train/',
                        help='train_path')
    parser.add_argument('--val_path',
                        default='data/val/',
                        help='val_path')
    parser.add_argument('--test_path',
                        default='data/test/',
                        help='test_path')
    # Mask parameters
    parser.add_argument('--accelerations',
                        nargs='+',
                        default=[4, 8],
                        type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for '
                             'each volume.')
    parser.add_argument('--center-fractions',
                        nargs='+',
                        default=[0.08, 0.04],
                        type=float,
                        help='Fraction of low-frequency k-space columns to be sampled. Should '
                             'have the same length as accelerations')
    ## model training
    parser.add_argument('--number_epoch',
                        type=int,
                        default=300,
                        help='num of training epoch')
    parser.add_argument('--val_on_epochs',
                        type=int,
                        default=1,
                        help='validate for each n epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='learning rate for training')
    parser.add_argument('--resume',
                        default='True',
                        action='store_true')

    args = parser.parse_args()

    print(args)
    solve(args)
