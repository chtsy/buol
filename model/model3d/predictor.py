# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Tao Chu
# ------------------------------------------------------------------------------

import torch
import MinkowskiEngine as Me

from collections import OrderedDict
from functools import partial
from torch import nn
from torch.nn import functional as F
from .aspp import ASPP
from .conv_module import stacked_conv
from utils.utils3d import sparse_cat_union
from utils.utils3d import mask_invalid_sparse_voxels


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
        upsample = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    Me.MinkowskiConvolution(low_level_channels[i], low_level_channels_project[i], 1, bias=False, dimension=3),
                    Me.MinkowskiInstanceNorm(low_level_channels_project[i]),
                    Me.MinkowskiReLU(inplace=True)
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
            if i == 0:
                up_in_channels = aspp_channels
            else:
                up_in_channels = decoder_channels
            upsample.append(
                Me.MinkowskiConvolutionTranspose(
                        up_in_channels, up_in_channels, kernel_size=4,
                        stride=2, bias=False, dimension=3, expand_coordinates=True
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)
        self.upsample = nn.ModuleList(upsample)

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
                Me.MinkowskiConvolution(head_channels, num_classes[i], 1, dimension=3),
            )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features, mask=None):
        x = features[self.feature_key]
        x = self.aspp(x)
        batch = x.C[:, 0].max().item() + 1

        if mask is None:
            coord_l = features[self.low_level_key[-1]].C.long()
            mask = torch.zeros((batch, 256, 256, 256), device=x.device, dtype=torch.bool)
            mask[coord_l[:, 0], coord_l[:, 1], coord_l[:, 2], coord_l[:, 3]] = True
            mask = {1:mask}
            for i in range(1, 4):
                mask.update({2**i:F.max_pool3d(mask[1].float().float(), i*2+1, 1, i).bool()})

        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = self.upsample[i](x)
            coord_x = x.C.long()
            valid_mask = (coord_x.max(-1)[0] < 256) & (coord_x.min(-1)[0] >= 0)
            coord_x = coord_x[valid_mask]
            mask_kept = mask[x.tensor_stride[-1]][coord_x[:, 0], coord_x[:, 1], coord_x[:, 2], coord_x[:, 3]]
            valid_mask[valid_mask.clone()] = mask_kept
            x = mask_invalid_sparse_voxels(x, valid_mask)
            x = sparse_cat_union(x, l)
            x = self.fuse[i](x)

        pred = OrderedDict()
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred
