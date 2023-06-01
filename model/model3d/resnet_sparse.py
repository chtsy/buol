import sys
from torch import nn
import MinkowskiEngine as Me

cur_mudule = sys.modules[__name__]
def backbone(cfg):
    return getattr(cur_mudule, cfg.NAME)(replace_stride_with_dilation=cfg.DILATION)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = Me.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = Me.MinkowskiInstanceNorm(planes)
        self.conv2 = Me.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = Me.MinkowskiInstanceNorm(planes)
        self.relu = Me.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0

        self.conv1 = Me.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = Me.MinkowskiInstanceNorm(planes)

        self.conv2 = Me.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = Me.MinkowskiInstanceNorm(planes)

        self.conv3 = Me.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = Me.MinkowskiInstanceNorm(
            planes * self.expansion)

        self.relu = Me.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Me.MinkowskiConvolution(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, dimension=3,)


class ResNetSparse(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetSparse, self).__init__()
        if norm_layer is None:
            norm_layer = Me.MinkowskiInstanceNorm
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.maxpool = Me.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=3)
        stride = 2
        downsample = nn.Sequential(
            Me.MinkowskiConvolution(13, self.inplanes, kernel_size=1, stride=stride, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(self.inplanes)
        )
        self.conv_sem = block(13, self.inplanes, stride=stride, downsample=downsample, dimension=3)
        downsample = nn.Sequential(
            Me.MinkowskiConvolution(2, self.inplanes, kernel_size=1, stride=stride, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(self.inplanes)
        )
        self.conv_dep = block(2, self.inplanes, stride=stride, downsample=downsample, dimension=3)
        downsample = nn.Sequential(
            Me.MinkowskiConvolution(1, self.inplanes, kernel_size=1, stride=stride, bias=True, dimension=3),
            Me.MinkowskiInstanceNorm(self.inplanes)
        )
        self.conv_occ = block(1, self.inplanes, stride=stride, downsample=downsample, dimension=3)
        self.inplanes = 2 * self.inplanes

        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, previous_dilation, downsample,
                            dimension=3))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.dilation,
                                dimension=3))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        outputs = {}
        cm = x.coordinate_manager
        key = x.coordinate_map_key
        x_sem = self.conv_sem(Me.SparseTensor(x.F[:, :13], coordinate_manager=cm, coordinate_map_key=key))
        x_occ = self.conv_occ(Me.SparseTensor(x.F[:, 13:14], coordinate_manager=cm, coordinate_map_key=key))
        x_dep = self.conv_dep(Me.SparseTensor(x.F[:, 14:], coordinate_manager=cm, coordinate_map_key=key))

        x = Me.cat(x_sem * x_occ, x_dep)
        outputs['stem'] = x

        x = self.layer1(x)  # 1/1
        outputs['res2'] = x

        x = self.layer2(x)  # 1/2
        outputs['res3'] = x

        x = self.layer3(x)  # 1/4
        outputs['res4'] = x

        x = self.layer4(x)  # 1/8
        outputs['res5'] = x

        return outputs

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNetSparse(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
