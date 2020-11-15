import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np


class Conv2d(nn.Conv2d):
    """Conv2d rewrites `nn.Conv2d`, adding the 'same' padding strategy
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 padding_method='valide'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bias, padding_mode)
        self.padding_method = padding_method

    def forward(self, input):
        if self.padding_method == 'valide':
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                return F.conv2d(F.pad(input, expanded_padding, mode='circular'), self.weight,
                                self.bias, self.stride, _pair(0), self.dilation, self.groups)
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation,
                            self.groups)
        elif self.padding_method == 'same':
            kernel_size = np.array(self.kernel_size)
            stride = np.array(self.stride)
            dilation = np.array(self.dilation)

            effective_kernel_size = (kernel_size - 1) * dilation + 1

            input_size = np.array([input.size(2), input.size(3)])
            output_size = (input_size + stride - 1) // stride

            padding_size = (output_size - 1) * stride + effective_kernel_size - input_size
            padding_size = np.where(padding_size > 0, padding_size, 0)
            expanded_padding = ((padding_size[1] + 1) // 2, padding_size[1] // 2,
                                (padding_size[0] + 1) // 2, padding_size[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'), self.weight, self.bias,
                            self.stride, _pair(0), self.dilation, self.groups)


if __name__ == '__main__':
    conv = Conv2d(3, 5, (3, 4), padding_method='same')
    x = torch.randn(10, 3, 50, 60)
    print(conv(x).size())
