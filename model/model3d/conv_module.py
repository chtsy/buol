# ------------------------------------------------------------------------------
# Common modules.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Tao Chu
# ------------------------------------------------------------------------------

import MinkowskiEngine as Me
from functools import partial
from torch import nn


def basic_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
               with_bn=True, with_relu=True):
    """convolution with bn and relu"""
    module = []
    has_bias = not with_bn
    module.append(
        Me.MinkowskiConvolution(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dimension=3,
                  bias=has_bias)
    )
    if with_bn:
        module.append(Me.MinkowskiInstanceNorm(out_planes))
    if with_relu:
        module.append(Me.MinkowskiReLU(inplace=True))
    return nn.Sequential(*module)


def depthwise_separable_conv(in_planes, out_planes, kernel_size, stride=1, padding=1, groups=1,
                             with_bn=True, with_relu=True):
    """depthwise separable convolution with bn and relu"""
    del groups

    module = []
    module.extend([
        basic_conv(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes,
                   with_bn=True, with_relu=True),
        Me.MinkowskiConvolution(in_planes, out_planes, kernel_size=1, stride=1, dimension=3, bias=False),
    ])
    if with_bn:
        module.append(Me.MinkowskiInstanceNorm(out_planes))
    if with_relu:
        module.append(Me.MinkowskiReLU(inplace=True))
    return nn.Sequential(*module)


def stacked_conv(in_planes, out_planes, kernel_size, num_stack, stride=1, padding=1, groups=1,
                 with_bn=True, with_relu=True, conv_type='basic_conv'):
    """stacked convolution with bn and relu"""
    if num_stack < 1:
        assert ValueError('`num_stack` has to be a positive integer.')
    if conv_type == 'basic_conv':
        conv = partial(basic_conv, out_planes=out_planes, kernel_size=kernel_size, stride=stride,
                       padding=padding, groups=groups, with_bn=with_bn, with_relu=with_relu)
    elif conv_type == 'depthwise_separable_conv':
        conv = partial(depthwise_separable_conv, out_planes=out_planes, kernel_size=kernel_size, stride=stride,
                       padding=padding, groups=1, with_bn=with_bn, with_relu=with_relu)
    else:
        raise ValueError('Unknown conv_type: {}'.format(conv_type))
    module = []
    module.append(conv(in_planes=in_planes))
    for n in range(1, num_stack):
        module.append(conv(in_planes=out_planes))
    return nn.Sequential(*module)
