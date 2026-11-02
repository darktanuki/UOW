"""
F = number of features
N = number of examples
O = number of outputs
Ch = number of channels
H = height
W = width
"""

import math
from typing import Sequence

from torch import Tensor
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU


def compute_conv_output_size(
    input_width: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    n_output_channels: int,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    width = compute_conv_output_width(input_width, kernel_sizes, strides, padding, dilation)

    return n_output_channels * (width**2)


def compute_conv_output_width(
    input_width: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """
    References:
        https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    width = input_width

    for kernel_size, stride in zip(kernel_sizes, strides):
        width = width + (2 * padding) - (dilation * (kernel_size - 1)) - 1
        width = math.floor((width / stride) + 1)

    return width


class ConvBlock(Module):
    def __init__(self, dropout_rate: float, n_in: int, n_out: int, kernel_size: int = 3) -> None:
        super().__init__()

        self.conv = Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=kernel_size)
        self.dropout = Dropout(p=dropout_rate)
        self.maxpool = MaxPool2d(kernel_size=2)
        self.activation_fn = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch_in, H_in, W_in]

        Returns:
            Tensor[float], [N, Ch_out, H_out, W_out]
        """
        x = self.conv(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.activation_fn(x)
        return x


class FullyConnectedBlock(Module):
    def __init__(self, dropout_rate: float, n_in: int, n_out: int) -> None:
        super().__init__()

        self.fc = Linear(in_features=n_in, out_features=n_out)
        self.dropout = Dropout(p=dropout_rate)
        self.activation_fn = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, F_in]

        Returns:
            Tensor[float], [N, F_out]
        """
        x = self.fc(x)
        x = self.dropout(x)
        x = self.activation_fn(x)
        return x


class ConvNet(Module):
    """
    References:
        https://github.com/BlackHC/batchbald_redux/blob/master/03_consistent_mc_dropout.ipynb
    """

    def __init__(
        self, c_in: int = 1, image_width: int = 32, dropout_rate: float = 0.0) -> None:
        super().__init__()

        n_input_channels = c_in
        image_width = 32

        fc1_size = compute_conv_output_size(
            image_width, kernel_sizes=(2 * (5, 2)), strides=(2 * (1, 2)), n_output_channels=64
        )

        self.block1 = ConvBlock(dropout_rate, n_in=n_input_channels, n_out=32, kernel_size=5)
        self.block2 = ConvBlock(dropout_rate, n_in=32, n_out=64, kernel_size=5)
        self.block3 = FullyConnectedBlock(dropout_rate, n_in=fc1_size, n_out=128)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch_in, H_in, W_in]

        Returns:
            Tensor[float], [N, O]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.block3(x)
        return x