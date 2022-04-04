import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_c: int, out_c: int, drop_prob: float):
        """
        :param in_c: input channels to the ConvBlock
        :param out_c: output channels to the ConvBlock
        :param drop_prob: Dropout probability
        """

        super(ConvBlock, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(self.in_c,
                      self.out_c,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.InstanceNorm2d(self.out_c),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.drop_prob),
            nn.Conv2d(self.out_c,
                      self.out_c,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.InstanceNorm2d(self.out_c),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(self.drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        :param image: Input 4D tensor of shape `(N, in_c, H, W)`
        :return: Output tensor of shape `(N, out_c, H, W)`
        """

        result = self.layers(image)
        return result


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution
    transpose layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_c: int, out_c: int):
        """
        :param in_c: Number of channels in the input.
        :param out_c: Number of channels in the output.
        """
        super(TransposeConvBlock, self).__init__()

        self.in_c = in_c
        self.out_c = out_c

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                self.in_c,
                self.out_c,
                kernel_size=2,
                stride=2,
                bias=False
            ),
            nn.InstanceNorm2d(self.out_c),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        :param image: Input 4D tensor of shape `(N, in_c, H, W)`.
        :return: Output tensor of shape `(N, out_c, H*2, W*2)`.
        """
        result = self.layers(image)
        return result


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model
    """

    def __init__(self, in_c, out_c, feats: int = 32, num_pool_layers: int = 4, drop_prob: float = 0.0):
        """
        :param in_c: input channels to the U-Net model
        :param out_c: output channels to the U-Net model
        :param feats: number of output channels of the first conv layers
        :param num_pool_layers: number of down-sampling and up-sampling layers
        :param drop_prob: Dropout probability
        :return:
        """
        super(Unet, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.feats = feats
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList(
            [ConvBlock(self.in_c, self.feats, self.drop_prob)]
        )
        tmp_ch = self.feats
        for _ in range(self.num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(
                tmp_ch,
                tmp_ch * 2,
                self.drop_prob
            ))
            tmp_ch *= 2

        self.conv = ConvBlock(tmp_ch, tmp_ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(
                tmp_ch * 2,
                tmp_ch
            ))
            self.up_conv.append(ConvBlock(
                tmp_ch * 2,
                tmp_ch,
                self.drop_prob
            ))
            tmp_ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(
            tmp_ch * 2,
            tmp_ch
        ))

        self.up_conv.append(
            nn.Sequential(
                ConvBlock(tmp_ch * 2,
                          tmp_ch,
                          self.drop_prob),
                nn.Conv2d(tmp_ch,
                          self.out_c,
                          kernel_size=1,
                          stride=1)
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        :param image: Input 4D tensor of shape `(N, in_c, H, W)`.
        :return: Output tensor of shape `(N, out_c, H, W)`.
        """

        stack = []
        output = image  # 1, 320, 320

        # Down sampling
        for layer in self.down_sample_layers:
            output = layer(output)  # 316 -> 154 -> 73 -> 32 -> 12
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)  # 158 -> 77 -> 36 -> 16 -> 6

        output = self.conv(output)

        # Up sampling
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/bottom if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


if __name__ == '__main__':
    model = Unet(1, 1).cuda()
    model(torch.rand((1, 1, 320, 320)).cuda())
