import sys

sys.path.append('../../')

from utils.net import conv_blocks
import torch
import torch.nn as nn


class DCLayer(nn.Module):
    """
    Create data consistency operator

    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DCLayer, self).__init__()
        self.norm = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, x_rec, k_un, mask):
        """
        :param x_rec: reconstructed input
        :param k_un: undersampled k-space
        :param mask: mask
        :param noise_level: noise_level
        """
        k_un, x_rec, mask = k_un.permute(0, 2, 3, 1), x_rec.permute(0, 2, 3, 1), mask.permute(0, 2, 3, 1)
        k_rec = torch.view_as_real(torch.fft.fft2(torch.view_as_complex(x_rec.contiguous()), norm=self.norm))

        # noiseless
        v = self.noise_level
        if not v:
            k_out = k_rec + (k_un - k_rec) * mask
        else:
            k_out = k_rec - mask * k_rec + mask * (k_rec + v * k_un) / (1 + v)

        x_out = torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(k_out), norm=self.norm))
        x_out = x_out.permute(0, 3, 1, 2)

        return x_out


class DCCNN(nn.Module):
    def __init__(self, n_iter=8, n_conv=6, n_filters=64, norm='ortho'):
        """
        DC-CNN
        :param n_iter: number of iterations
        :param n_convs: number of convs in each block
        :param n_filters: number of feature channels in intermediate features
        :param norm: 'ortho' norm for fft
        """

        super(DCCNN, self).__init__()
        channel_in = 2
        rec_blocks = []
        dcs = []
        self.norm = norm
        self.mu = nn.Parameter(torch.Tensor([0.5]))
        self.n_iter = n_iter

        for i in range(n_iter):
            rec_blocks.append(conv_blocks('dccnn', channel_in, n_filters=n_filters, n_conv=n_conv))
            dcs.append(DCLayer(norm='ortho'))

        self.rec_blocks = nn.ModuleList(rec_blocks)
        self.dcs = dcs

    def _forward_operation(self, img, mask):
        k = torch.fft.fft2(torch.view_as_complex(img.permute(0, 2, 3, 1).contiguous()),
                           norm=self.norm)
        k = torch.view_as_real(k).permute(0, 3, 1, 2).contiguous()
        k = mask * k
        return k

    def _backward_operation(self, k, mask):
        k = mask * k
        img = torch.fft.ifft2(torch.view_as_complex(k.permute(0, 2, 3, 1).contiguous()),
                              norm=self.norm)
        img = torch.view_as_real(img).permute(0, 3, 1, 2).contiguous()
        return img

    def update_operation(self, f_1, k, mask):
        h_1 = k - self._forward_operation(f_1, mask)
        update = f_1 + self.mu * self._backward_operation(h_1, mask)
        return update

    def forward(self, x, k, m):
        for i in range(self.n_iter):
            x_cnn = self.rec_blocks[i](x)
            x += x_cnn
            x = self.update_operation(x, k, m)
            #x = self.dcs[i](x, k, m)

        return x
