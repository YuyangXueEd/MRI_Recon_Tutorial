import torch
import torch.nn as nn
from collections import OrderedDict

def conv_blocks(model_name='dccnn', channel_in=22, n_conv=3, n_filters=32):
    """
    reconstruction blocks in DC-CNN;
    :param model_name: 'dc-cnn', 'prim-net', 'dual-net', or 'hqs-net'
    :param channel_in: input channels
    :param n_convs: how much convs
    :param n_filters: feature size
    :return:
    """

    layers = []
    if model_name == 'dccnn':
        channel_out = channel_in
    else:
        raise(ValueError("No such model."))

    for i in range(n_conv - 1):
        if i == 0:
            layers.append(nn.Conv2d(channel_in, n_filters, kernel_size=3, stride=1, padding=1))
        else:
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers.append(nn.Conv2d(n_filters, channel_out, kernel_size=3, stride=1, padding=1))

    return nn.Sequential(*layers)

