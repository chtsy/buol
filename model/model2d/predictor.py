# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Tao Chu
# ------------------------------------------------------------------------------

import torch
from collections import OrderedDict
from functools import partial
from torch import nn
from torch.nn import functional as F
from .aspp import ASPP
from .conv_module import stacked_conv


class Predictor(nn.Module):
    def __init__(self, cfg_feat, cfg_pred):
        super(Predictor, self).__init__()
        aspp_channels = cfg_pred.ASPP.ASPP_CHANNELS
        decoder_channels = cfg_pred.DECODER.DECODER_CHANNELS
        self.aspp = ASPP(
            cfg_feat.IN_CHANNELS,
            out_channels=aspp_channels,
            atrous_rates=cfg_pred.ASPP.ATROUS_RATES)
        self.feature_key = cfg_feat.FEATURE_KEY
        self.decoder_stage = len(cfg_feat.LOW_LEVEL_CHANNELS)
        self.low_level_key = cfg_feat.LOW_LEVEL_KEY
        low_level_channels = cfg_feat.LOW_LEVEL_CHANNELS
        low_level_channels_project = cfg_pred.ASPP.LOW_LEVEL_CHANNELS_PROJECT
        assert self.decoder_stage == len(self.low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(low_level_channels[i], low_level_channels_project[i], 1, bias=False),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU()
                )
            )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_conv(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

        head_channels = cfg_pred.HEAD.HEAD_CHANNELS
        num_classes = cfg_pred.HEAD.NUM_CLASSES
        class_key = cfg_pred.HEAD.CLASS_KEY

        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                fuse_conv(
                    decoder_channels,
                    head_channels,
                ),
                nn.Conv2d(head_channels, num_classes[i], 1)
            )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        x = features[self.feature_key]
        x = self.aspp(x)

        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)

        pred = OrderedDict()
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred

