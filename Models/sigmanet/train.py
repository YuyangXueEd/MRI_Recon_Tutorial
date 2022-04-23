import sys
import functools
import logging
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard.writer import SummaryWriter

sys.path.append('../../')
from Datasets.FastMRI.args import TrainArgs
import Datasets.FastMRI.multicoil as data_mc
from Datasets.medutils_torch import loss
from utils.template import build_optim, define_losses, postprocess, save_image_writer, save_model,
from modules.datalayer import DataIDLayer, DataGDLayer, DataProxCGLayer, DataVSLayer
from collections.unet_baseline import UnetModel
from collections.sn import SensitivityNetwork
from collections.pcn import ParallelCoilNetwork


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_loss(args):
    losses = define_losses()
    l1 = losses['l1']
    ssim = losses['ssim']

    def criterion(output, target, sample, **kwargs):
        # if there is no key 'scale', return 1.
        scale = kwargs.pop('scale', 1.)
        loss_l1 = l1(output, target, sample)
        loss_ssim = ssim(output, target, sample)

        # according to the paper
        # The parameter \gamma_{l_1}= 1e−5 is chosen empirically
        # to match the scale of the two losses and is motivated by
        # the fastMRI challenge requirements.
        loss = loss_ssim + loss_l1 * 1e-3
        loss /= scale

        return loss, loss_l1, loss_ssim

    return criterion


def create_batch_datasets(args, **kwargs):
    """
    Create dataset based on csv files
    :param args: arguments list
    :return : batch dataset
    """

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dataset
    csv_train = kwargs.pop('csv_train', f'{args.csv_path}/multicoil_train.csv')
    csv_val = kwargs.pop('csv_val', f'{args.csv_path}/multicoil_val.csv')

    data_filter = kwargs.pop('data_filter', {})
    train_slices = kwargs.pop('train_slices', {'min': 5})
    test_slices = kwargs.pop('test_slices', {'min': 12, 'max': 25})

    # transforms
    train_transforms = [
        data_mc.GenerateRandomFastMRIChallengeMask(
            args.center_fractions,
            args.accelerations,
            is_train=True
        ),
        data_mc.LoadCoilSensitivities(
            'multicoil_train_espirit',
            num_smaps=args.nsmaps,
        ),
        data_mc.ComputeBackgroundNormalization(),
        data_mc.LoadForegroundMask('multicoil_train_foreground') if args.use_fg_mask else data_mc.SetupForegroundMask(),
        data_mc.GeneratePatches(args.fe_patch_size),
        data_mc.ComputeInit(args.pcn),
        data_mc.ToTensor()
    ]

    val_transforms = [
        data_mc.GenerateRandomFastMRIChallengeMask(
            args.center_fractions,
            args.accelerations,
            is_train=False
        ),
        data_mc.LoadCoilSensitivities(
            'multicoil_val_espirit',
            num_smaps=args.nsmaps
        ),
        data_mc.ComputeBackgroundNormlization(),
        data_mc.LoadForegroundMask('multicoil_val_foreground') if args.use_fg_mask else data_mc.SetupForegroundMask(),
        data_mc.GeneratePatches(320),
        data_mc.ComputeInit(args.pcn),
        data_mc.ToTensor()
    ]

    train_dataset = data_mc.MRIDataset(
        csv_train,
        args.data_path,
        transform=transforms.Compose(train_transforms),
        batch_size = args.batch_size,
        slices=train_slices,
        data_filter=data_filter,
        norm=args.norm,
        full=args.full_slices
    )

    if len(train_dataset) == 0:
        raise ValueError('Train dataset has length 0.')

    val_dataset = data_mc.MRIDataset(
        csv_val,
        args.data_path,
        transform=transforms.Compose(val_transforms),
        batch_size=args.batch_size,
        slices=test_slices,
        data_filter=data_filter,
        norm=args.norm
    )

    if len(val_dataset) == 0:
        raise ValueError('Val dataset has length 0.')

    return train_dataset, val_dataset


def create_data_loaders(args, **kwargs):
    """
    Create data loaders
    """

    train_data, val_data = create_batch_datasets(args, **kwargs)
    display_data = [val_data[0]]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=1,
        pin_memory=False
    )

    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        num_workers=1,
        pin_memory=False
    )

    return train_loader, val_loader, display_loader


def train_epoch(state, model, data_loader, optimizer, writer):
    # set to train mode
    model.train()

    args = state.args
    keys = ['input', 'target', 'kspace', 'smaps', 'mask', 'fg_mask']
    attr_keys = ['mean', 'cov', 'ref_max']
    save_image = functools.partial(save_image_writer, writer, state.epoch)

    # init average loss
    avg_loss = 0.
    # start
    start_epoch = start_iter = time.perf_counter()

    # init average performance
    perf_avg = 0
    for iter, sample in enumerate(data_loader):
        # init load data time
        t0 = time.perf_counter()
        sample = data_mc._read_data(sample, keys, attr_keys, args.device)
        output = model(
            sample['input'],
            sample['kspace'],
            sample['smaps'],
            sample['mask'],
            sample['attrs']
        )

        # get recon area
        rec_x = sample['attr']['metadata']['rec_x']
        rec_y = sample['attr']['metadata']['rec_y']

        # crop
        output = postprocess(output, (rec_x, rec_y))
        target = postprocess(sample['target'], (rec_x, rec_y))
        sample['fg_mask'] = postprocess(sample['fg_mask'], (rec_x, rec_y))

        # get loss func
        loss, loss_l1, loss_ssim = state.loss_fn(
            output=output,
            target=target,
            sample=sample,
            scale= 1. / args.grad_acc
        )

        # calculate loss; forward time
        t1 = time.perf_counter()
        loss.backward()
        if state.gloabl_step % state.grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()

        state.global_step += 1
        # calculate performance; backward time
        t2 = time.perf_counter()
        perf = t2 - start_iter
        perf_avg += perf

        avg_loss = 0.99 * avg_loss + (0.01 if iter > 0 else 1) * loss.item()

        # logging
        if iter % args.report_interval == 0:
            writer.add_scalar('Train Loss', loss.item(), state.global_step)
            writer.add_scalar('Train L1 Loss', loss_l1.item(), state.global_step)
            writer.add_scalar('Train SSIM Loss', loss_ssim.item(), state.global_step)
            logging.info(
                f'Epoch = [{state.epoch:3d}/{args.num_epochs:3d}]\t'
                f'Iter = [{iter:4d}/{len(data_loader):4d}]\t'
                f'Loss = {loss.item():.4g}\t Avg Loss = {avg_loss:.4g}\t'
                f't = (total{perf_avg / (iter + 1):.1g}s\t'
                f'fwd: {t1-t0:.1g}s\t bwd: {t2-t1:.1g}s\t'
            )

        if state.gloabal_step % 1000 == 0:
            input_abs = postprocess(sample['input'], (rec_x, rec_y))
            base_err = torch.abs(target - input_abs)
            pred_err = torch.abs(target - output)
            residual = torch.abs(input_abs - output)

            save_image(
                torch.cat([input_abs, output, target], -1).unsqueeze(0),
                'Train_undersampled_pred_gt',
            )
            save_image(
                torch.cat([base_err, pred_err, residual], -1).unsqueeze(0),
                'Train_Err_base_pred_res',
                base_err.max(),
            )

            save_model(args, args.outdir, state.epoch, model, optimizer,
                       avg_loss, is_new_best=False, modelname='model_tmp.pt')

        start_iter = time.perf_counter()

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(state, model, data_loader, metrics, writer):
    # eval mode
    model.eval()
    keys = ['input', 'target', 'kspace', 'smaps', 'mask', 'fg_mask']
    attr_keys = ['mean', 'conv', 'ref_max']
    losses = defaultdict(list)

    start = time.perf_counter()
    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            sample = data_mc._read_data(sample, keys, attr_keys, args.device)
            output = model(
                sample['input'],
                sample['kspace'],
                sample['smaps'],
                sample['mask'],
                sample['attrs']
            )

            rec_x = sample['attrs']['metadata']['rec_x']
            rec_y = sample['attrs']['metadata']['rec_y']

            output = postprocess(output, (rec_x, rec_y))
            target = postprocess(sample['target'], (rec_x, rec_y))
            sample['fg_mask'] = postprocess(sample['fg_mask'], (rec_x, rec_y))
            loss = state.loss_fn(
                output=output,
                target=target,
                sample=sample,
                scale=1. / state.grad_acc,
            )[0]
            losses['dev_loss'].append(loss.item())

            # evaluate in the foreground
            target = target.unsqueeze(1) * sample['fg_mask']
            output = output.unsqueeze(1) * sample['fg_mask']
            for k in metrics:
                losses[k].append(metrics[k](target, output, sample).item())

        for k in losses:
            writer.add_scalar(f'Val_{k}', np.mean(losses[k]), state.epoch)

    return losses, time.perf_counter() - start

def visualize(state, model, data_loader, writer):
    



