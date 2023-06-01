# Hu et al. Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries
# Original code from: https://github.com/JunjH/Revisiting_Single_Depth_Estimation
# Modified by Tao Chu

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from ..loss import loss


class _UpProjection(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)

        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out


class DepthPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        block_channel = cfg.BLOCK_CHANNELS
        self.feature_key = cfg.FEATURE_KEY
        self.class_key = cfg.CLASS_KEY
        self.prediction = {}
        self.get_gradient = Sobel()  # .to(config.MODEL.DEVICE)

        self.D = D(block_channel)
        self.MFF = MFF(block_channel, cfg.FEATURE_CHANNELS)
        C = []
        head_channels = block_channel[-1] // 4 + cfg.FEATURE_CHANNELS
        for name, channel in zip(cfg.CLASS_KEY, cfg.NUM_CLASSES):
            self.prediction[name] = channel
            C.append(R(head_channels, channel))
        self.C = nn.ModuleList(C)

        self.cos_loss = nn.CosineSimilarity(dim=1, eps=0)
        self.depth_offset = -np.log(.5)
        self.loss_depth_weight = cfg.LOSS.DEPTH.WEIGHT
        self.loss_occupancy = loss(cfg.LOSS.OCCUPANCY2D)
        self.loss_occupancy_weight = cfg.LOSS.OCCUPANCY2D.WEIGHT

    def forward(self, x, target=None):
        x_block1, x_block2, x_block3, x_block4 =\
            x[self.feature_key[3]], x[self.feature_key[2]], x[self.feature_key[1]], x[self.feature_key[0]]
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
        x_feat = torch.cat((x_decoder, x_mff), 1)
        out = {}
        for k, ci in zip(self.class_key, self.C):
            out[k] = ci(x_feat)

        if target is not None:
            losses = self.loss(out, target['depth'])
        else:
            losses = 0

        return out, losses

    def loss(self, pred, target):
        eps = 1e-7
        losses = self.loss_depth(pred['depth'], target['depth_map'])

        loss_occ = self.loss_occupancy(
            pred['occupancy2d'].sigmoid(), target['occupancy'])
        valid_masks = torch.stack([(depth != 0.0).bool() for depth in target['depth_map']], dim=0)
        loss_occ = loss_occ * valid_masks[:, None]
        losses['Occupancy2D'] = (loss_occ[target['occupancy_weight']]).sum() / (
                    valid_masks.sum() + eps) * self.loss_occupancy_weight

        return losses

    def loss_depth(self, pred, target, losses={}):
        device = pred.device
        target = target[:, None]
        valid_masks = torch.stack([(depth != 0.0).bool() for depth in target], dim=0)

        grad_target = self.get_gradient(target)
        grad_pred = self.get_gradient(pred)

        grad_target_dx = grad_target[:, 0, :, :].contiguous().view_as(target)
        grad_target_dy = grad_target[:, 1, :, :].contiguous().view_as(target)
        grad_pred_dx = grad_pred[:, 0, :, :].contiguous().view_as(target)
        grad_pred_dy = grad_pred[:, 1, :, :].contiguous().view_as(target)

        ones = torch.ones(target.size(0), 1, target.size(2), target.size(3)).float().to(
            device)
        normal_target = torch.cat((-grad_target_dx, -grad_target_dy, ones), 1)
        normal_pred = torch.cat((-grad_pred_dx, -grad_pred_dy, ones), 1)

        loss_depth = torch.log(torch.abs(target - pred) + 0.5)[
                         valid_masks].mean() + self.depth_offset  ## fix
        loss_dx = torch.log(torch.abs(grad_target_dx - grad_pred_dx) + 0.5)[valid_masks].mean() + self.depth_offset
        loss_dy = torch.log(torch.abs(grad_target_dy - grad_pred_dy) + 0.5)[valid_masks].mean() + self.depth_offset
        loss_normal = torch.abs(1 - self.cos_loss(normal_pred, normal_target))[valid_masks.squeeze(1)].mean()
        loss_gradient = loss_dx + loss_dy

        losses['Depth'] = loss_depth * self.loss_depth_weight
        losses['Normal'] = loss_normal * self.loss_depth_weight
        losses['Gradient'] = loss_gradient * self.loss_depth_weight

        return losses

class D(nn.Module):
    def __init__(self, block_channel):
        super().__init__()
        self.conv = nn.Conv2d(block_channel[0], block_channel[1], kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(block_channel[1])

        self.up1 = _UpProjection(num_input_features=block_channel[1],
                                 num_output_features=block_channel[2])

        self.up2 = _UpProjection(num_input_features=block_channel[2],
                                 num_output_features=block_channel[3])

        add_feat_channel = block_channel[3]
        self.up3 = _UpProjection(num_input_features=add_feat_channel,
                                 num_output_features=add_feat_channel // 2)

        add_feat_channel = add_feat_channel // 2
        self.up4 = _UpProjection(num_input_features=add_feat_channel,
                                 num_output_features=add_feat_channel // 2)

    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3, [x_block1.size(2) * 2, x_block1.size(3) * 2])

        return x_d4


class MFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super().__init__()
        self.up1 = _UpProjection(num_input_features=block_channel[3], num_output_features=16)
        self.up2 = _UpProjection(num_input_features=block_channel[2], num_output_features=16)
        self.up3 = _UpProjection(num_input_features=block_channel[1], num_output_features=16)
        self.up4 = _UpProjection(num_input_features=block_channel[0], num_output_features=16)

        self.conv = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x


class R(nn.Module):
    def __init__(self, channel, num_class=1):
        super().__init__()
        self.conv0 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(channel)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(channel, num_class, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)

        return x2


class Sobel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out
