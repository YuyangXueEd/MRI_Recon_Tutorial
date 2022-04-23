import torch
import torch.nn.functional as F


def build_metrics(args):
    _ssim = SSIM(device=args.device)
    _psnr = PSNRLoss()
    metrics = {
        'MSE': lambda x, y, kwargs: F.mse_loss(x, y),
        'NMSE': lambda x, y, kwargs: nmse(x,y),
        'PSNR': lambda x, y, s: _psnr(x, y, data_range=s['attrs']['ref_max'])
        'SSIM': lambda x, y, s: _ssim(x, y, data_range=s['attrs']['ref_max'])
    }

    return metrics