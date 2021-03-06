import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


def _fspecial_gauss_1d(size, sigma):
    """
    Create 1D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    """
    Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        win (torch.Tensor): 1-D gauss kernel
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise ValueError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}")

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    """
    Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1D gauss kernel
        data_range (float or int): value range of input iamge (1.0 or 255)
        size_average (bool): if size_average=True, sim of all images will be averaged as a scalar

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


class SSIM(nn.Module):
    pass


def ms_ssim(X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5,
            win=None, weights=None, K=(0.01, 0.03)):
    """ interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """

    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise TypeError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4d or 5d tensors, but got {X.shape}")

    if win is not None:
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
                2 ** 4), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % (
                (win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class MS_SSIM(nn.Module):
    def __init__(self, size_average=True, win_size=11, win_sigma=1.5,
                 channel=3, spatial_dims=2, weights=None, K=(0.01, 0.03)):
        """
        Args:

        :params data_range: (float or int), value range of input image (1.0 or 255)
        :params size_average: (bool), if `size_average=True`, ssim of all images will be averaged as a scalar
        :params win_size: (int), the size of gaussian kernel
        :params win_sigma: (float), sigma of normal distribution
        :params channel: (int), input channels
        :params weights: (list), weights for different levels
        :params K: (list or tuple), scalar constants (K1, K2).
        """
        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.weights = weights
        self.K = K

    def forward(self, x, y, data_range=1.):
        return ms_ssim(x, y, data_range=data_range,
                       size_average=self.size_average,
                       win=self.win,
                       win_size=self.win_size,
                       weights=self.weights,
                       K=self.K)


class CompoundLoss(nn.Module):
    def __init__(self, ssim_type='ssim'):
        super(CompoundLoss, self).__init__()
        self.l1loss = nn.L1Loss()
        if ssim_type == 'ssim':
            self.ssim = SSIM(win_size=7, size_average=True, channel=1, K=(0.01, 0.03))
        elif ssim_type == 'ms-ssim':
            self.ssim = MS_SSIM(win_size=7, size_average=True, channel=1, K=(0.01, 0.03))
        self.alpha = 0.84

    def forward(self, pred, target, data_range=1.):
        l1_loss = self.l1loss(pred, target)
        ssim_loss = 1 - self.ssim(pred.unsqueeze(1), target.unsqueeze(1), data_range)
        return (1 - self.alpha) * l1_loss + self.alpha * ssim_loss
