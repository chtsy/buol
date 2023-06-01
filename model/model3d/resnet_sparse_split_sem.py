from torch import nn
import MinkowskiEngine as Me
from lib_3d.utils import sparse_cat_union


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
        # downsample = nn.Sequential(
        #     Me.MinkowskiConvolution(2, self.inplanes, kernel_size=1, stride=stride, bias=True, dimension=3),
        #     Me.MinkowskiInstanceNorm(self.inplanes)
        # )
        # self.conv_ins = block(2, self.inplanes, stride=stride, downsample=downsample, dimension=3)
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
        ## self.conv_fuse = block(3*self.inplanes, 2*self.inplanes, downsample=downsample, dimension=3)
        # self.inplanes = 3 * self.inplanes
        # self.conv_sem = nn.Sequential(
        #     Me.MinkowskiConvolution(13, self.inplanes, kernel_size=7, stride=2, dimension=3,
        #                             bias=False),
        #     norm_layer(self.inplanes),
        #     Me.MinkowskiReLU(inplace=True)
        # )
        # self.conv_ins = nn.Sequential(
        #     Me.MinkowskiConvolution(2, self.inplanes, kernel_size=7, stride=2, dimension=3,
        #                             bias=False),
        #     norm_layer(self.inplanes),
        #     Me.MinkowskiReLU(inplace=True)
        # )
        # self.conv_dep = nn.Sequential(
        #     Me.MinkowskiConvolution(2, self.inplanes, kernel_size=7, stride=2, dimension=3,
        #                             bias=False),
        #     norm_layer(self.inplanes),
        #     Me.MinkowskiReLU(inplace=True)
        # )
        self.inplanes = 2 * self.inplanes

        ## self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        #### need fix
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        '''

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
        # See note [TorchScript super()]
        # x = self.maxpool(x)
        cm = x.coordinate_manager
        key = x.coordinate_map_key
        x_sem = self.conv_sem(
            Me.SparseTensor(
                x.F[:, :13], coordinate_manager=cm, coordinate_map_key=key))
        # x_ins = self.conv_ins(
        #     Me.SparseTensor(
        #         x.F[:, 13:15], coordinate_manager=cm, coordinate_map_key=key))
        x_occ = self.conv_occ(
            Me.SparseTensor(
                x.F[:, 13:14], coordinate_manager=cm, coordinate_map_key=key))
        x_dep = self.conv_dep(
            Me.SparseTensor(
                x.F[:, 14:], coordinate_manager=cm, coordinate_map_key=key))
        # x = Me.cat(x_sem, x_ins)
        # x = Me.cat(x, x_dep)
        ## x = self.conv_fuse(x)
        ## x = self.maxpool(x)
        # x = x_sem * x_dep
        x = Me.cat(x_sem * x_occ, x_dep) ## 3
        # x = Me.cat(x_sem, x_dep) ## baseline 30
        # x = Me.cat(x_sem, x_occ, x_dep) ## 31
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


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetSparse(block, layers, **kwargs)
    #### need fix
    '''
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    '''
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
