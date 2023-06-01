# ------------------------------------------------------------------------------
# Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
# Modified by Tao Chu
# ------------------------------------------------------------------------------

from torch import nn
import MinkowskiEngine as Me
from utils.utils3d import sparse_cat_union


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            Me.MinkowskiConvolution(in_channels, out_channels, 3, dilation=dilation,
                                    bias=False, dimension=3),
            Me.MinkowskiInstanceNorm(out_channels),
            Me.MinkowskiReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.aspp_pooling = nn.Sequential(
            Me.MinkowskiGlobalAvgPooling(), ##nn.AdaptiveAvgPool2d(1),
            Me.MinkowskiConvolution(in_channels, out_channels, 1, bias=False, dimension=3),
            Me.MinkowskiReLU(inplace=True)
        )

    def set_image_pooling(self, pool_size=None):
        if pool_size is None:
            self.aspp_pooling[0] = Me.MinkowskiGlobalAvgPooling() ##nn.AdaptiveAvgPool2d(1)
        else:
            self.aspp_pooling[0] = Me.MinkowskiAvgPooling(kernel_size=pool_size, stride=1)

    def forward(self, x):
        x_p = self.aspp_pooling(x)
        feats = x_p.F[x.C[:, 0].long()]
        x = Me.SparseTensor(
            features=feats,
            coordinates=x.C,
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
        )
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            Me.MinkowskiConvolution(in_channels, out_channels, 1, bias=False, dimension=3),
            Me.MinkowskiInstanceNorm(out_channels),
            Me.MinkowskiReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            Me.MinkowskiConvolution(5 * out_channels, out_channels, 1, bias=False, dimension=3),
            Me.MinkowskiInstanceNorm(out_channels),
            Me.MinkowskiReLU(inplace=True),
            Me.MinkowskiDropout(0.5))

    def set_image_pooling(self, pool_size):
        self.convs[-1].set_image_pooling(pool_size)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = sparse_cat_union(Me.cat(res[:-1]), res[-1])
        return self.project(res)
