import numpy as np
import torch
import torch.nn.functional as F

def center_crop(data, shape):
    """
    Apply a center crop to the input real image (batches)

    :params data -> torch.tensor: The input tensor to be center cropped.
                  It should have at least 2 dimensions and the cropping is
                  applied along the last two dimensions.
    :params shape -> (int, int): The output shape. The shape should be
                  smaller than the corresponding dimensions of data.
    :return data -> torch.tensor: the center cropped image
    """

    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]