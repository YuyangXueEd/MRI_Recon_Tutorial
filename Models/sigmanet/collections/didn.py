import sys

sys.path('../../../')

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.net import calculate_downsampling_padding2d, pad2d, unpad2d


class Res_Block(nn.Module):
    def __init__(self, feats=64):
        super(Res_Block, self).__init__()

        bias = True

        # res1 (x)
        self.conv1 = nn.Conv2d(feats, feats, 3, 1, 1, bias)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(feats, feats, 3, 1, 1, bias)
        self.relu2 = nn.PReLU()
        # cat1

        self.conv3 = nn.Conv2d(feats, feats * 2, 3, 2, 1, bias)
        self.relu3 = nn.PReLU()
        # res2
        self.conv4 = nn.Conv2d(feats * 2, feats * 2, 3, 1, 1, bias)
        self.relu4 = nn.PReLU()
        # cat2

        self.conv5 = nn.Conv2d(feats * 2, feats * 4, 3, 2, 1, bias)
        self.relu5 = nn.PReLU()
        # res3
        self.conv6 = nn.Conv2d(feats * 4, feats * 4, 3, 1, 1, bias)
        self.relu6 = nn.PReLU()
        # (cat3)

        self.conv7 = nn.Conv2d(feats * 4, feats * 8, 1, 1, 0, bias)
        self.up7 = nn.PixelShuffle(2)

        # cat(cat2)
        self.conv8 = nn.Conv2d(feats * 4, feats * 2, 1, 1, 0, bias)
        # res4
        self.conv9 = nn.Conv2d(feats * 2, feats * 2, 3, 1, 1, bias)
        self.relu9 = nn.PReLU()
        # (cat4)

        self.conv10 = nn.Conv2d(feats * 2, feats * 4, 1, 1, 0, bias)
        self.up10 = nn.PixelShuffle(2)
        # cat(cat1)

        self.conv11 = nn.Conv2d(feats * 2, feats, 1, 1, 0, bias)
        # res5
        self.conv12 = nn.Conv2d(feats, feats, 3, 1, 1, bias)
        self.relu12 = nn.PReLU()
        self.conv13 = nn.Conv2d(feats, feats, 3, 1, 1, bias)
        self.relu13 = nn.PReLU()
        # cat(res5)

        self.conv14 = nn.Conv2d(feats, feats, 3, 1, 1, bias)

    @staticmethod
    def crcr(x, conv1, relu1, conv2, relu2, need_res=True):
        if need_res:
            res = relu1(conv1(x))
            return res + relu2(conv2(res))
        else:
            return x + relu2(conv2(relu1(conv1)))

    @staticmethod
    def conv_up(x, conv, up):
        return up(conv(x))

    def forward(self, x):
        res1 = x

        cat1 = torch.add(res1, self.crcr(x, self.conv1, self.relu1, self.conv2, self.relu2, False))
        cat2 = self.crcr(cat1, self.conv3, self.relu3, self.conv4, self.relu4, True)
        cat3 = self.crcr(cat2, self.conv5, self.relu4, self.conv6, self.relu6, True)

        catcat2 = torch.add(cat2, self.conv_up(cat3, self.conv7, self.up7))

        res4 = self.conv8(catcat2)
        cat4 = torch.add(res4, self.relu9(self.conv9(res4)))

        catcat1 = torch.add(cat1, self.conv_up(cat4, self.conv10, self.up10))

        res5 = self.conv11(catcat1)
        cat5 = self.crcr(res5, self.conv12, self.relu12, self.conv13, self.relu13, False)

        return torch.add(res1, self.conv14(cat5))


class Recon_Block(nn.Module):

    def __init__(self, feats=64, n_module=4):
        super(Recon_Block, self).__init__()
        bias = True

        for i in range(n_module):
            setattr(self, 'conv' + str(i * 2 - 1), nn.Conv2d(feats, feats, 3, 1, 1, bias))
            setattr(self, 'relu' + str(i * 2 - 1), nn.PReLU())
            setattr(self, 'conv' + str(i * 2), nn.Conv2d(feats, feats, 3, 1, 1, bias))
            setattr(self, 'relu' + str(i * 2), nn.PReLU())

        setattr(self, str(i * n_module + 1), nn.Conv2d(feats, feats, 3, 1, 1, bias))

    @staticmethod
    def res_block(x, conv1, relu1, conv2, relu2):
        return x + relu2(conv2(relu1(conv1(x))))

    def forward(self, x):
        res1 = x
        out = torch.add(res1, self.res_block(x, self.conv1, self.relu1, self.conv2, self.relu2))

        res2 = out
        out = torch.add(res2, self.res_block(x, self.conv3, self.relu3, self.conv4, self.relu4))

        res3 = out
        out = torch.add(res3, self.res_block(x, self.conv5, self.relu5, self.conv6, self.relu6))

        res4 = out
        out = torch.add(res4, self.res_block(x, self.conv7, self.relu7, self.conv8, self.relu8))

        return torch.add(res1, self.conv9(out))


class DIDN(nn.Module):
    """
    Deep Iterative Down-up Network, NRIRE denosing challenge winning entry
    Source: http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yu_Deep_Iterative_Down-Up_CNN_for_Image_Denoising_CVPRW_2019_paper.pdfp

    """

    def __init__(self, in_c, out_c, feats=64, pad_data=True, global_residual=True, n_res_blocks=6):
        super(DIDN, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.feats = feats
        self.pad_data = pad_data
        self.global_residual = global_residual
        self.n_res_blocks = n_res_blocks
        self.bias = True

        self.conv_input = nn.Conv2d(self.in_c, self.feats, 3, 1, 1, self.bias)
        self.relu1 = nn.PReLU()
        self.conv_down = nn.Conv2d(self.feats, self.feats, 3, 2, 1, self.bias)
        self.relu2 = nn.PReLU()

        recursive = []
        for i in range(self.n_res_blocks):
            recursive.append(Res_Block(self.feats ))

        self.recursive = torch.nn.ModuleList(recursive)

        self.conv_mid = nn.Conv2d(self.feats  * self.n_res_blocks, self.feats , 1, 1, 0, self.bias)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(self.feats , self.feats , 3, 1, 1, self.bias)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(self.feats, self.out_c, 3, 1, 1, self.bias)

    def forward(self, x):
        if self.pad_data:
            orig_shape2d  = x.shape[-2:]
            p2d = calculate_downsampling_padding2d(x, 3)
            x = pad2d(x, p2d)

        res1 = x
        out = self.relu2(self.conv_down(self.relu1(self.conv_input(x))))

        recons = []
        for i in range(self.n_res_blocks):
            out = self.recursive[i](out)
            recons.append(out)

        out = torch.cat(recons, 1)

        out = self.relu3(self.conv_mid(out))
        res2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, res2)

        out = self.conv_output(self.subpixel(out))

        if self.global_residual:
            out = torch.add(out, res1)

        if self.pad_data:
            out = unpad2d(out, orig_shape2d)

        return out


