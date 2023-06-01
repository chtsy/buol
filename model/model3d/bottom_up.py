import torch
from torch import nn
import torch.nn.functional as F
from .predictor import Predictor
from utils.utils3d import mask_invalid_sparse_voxels
from utils.dense_layer import dense_layer
from ..loss import loss

class BottomUp3D(nn.Module):
    def __init__(self, cfg):
        super(BottomUp3D, self).__init__()
        self.semantic = Predictor(cfg.IN_FEATURE, cfg.SEMANTIC)
        self.instance = Predictor(cfg.IN_FEATURE, cfg.INSTANCE)
        self.geometry = Predictor(cfg.IN_FEATURE, cfg.GEOMETRY)

        self.grid_dimensions = cfg.GRID_DIMENSIONS
        self.truncation = cfg.TRUNCATION

        self.loss_occupancy = loss(cfg.LOSS.OCCUPANCY)
        self.loss_semantic = loss(cfg.LOSS.SEMANTIC)
        self.loss_center = loss(cfg.LOSS.CENTER)
        self.loss_offset = loss(cfg.LOSS.OFFSET)
        self.loss_geometry = loss(cfg.LOSS.GEOMETRY)

        self.loss_weight_occupancy = cfg.LOSS.OCCUPANCY.WEIGHT
        self.loss_weight_semantic = cfg.LOSS.SEMANTIC.WEIGHT
        self.loss_weight_center = cfg.LOSS.CENTER.WEIGHT
        self.loss_weight_offset = cfg.LOSS.OFFSET.WEIGHT
        self.loss_weight_geometry = cfg.LOSS.GEOMETRY.WEIGHT

    def forward(self, features, mask, frustum, target=None):
        mask = mask * frustum
        mask_stride = {1: mask}
        mask_stride.update({2**i: frustum * F.max_pool3d(mask.float(), 2**i+1, 1, 2**(i-1)).bool() for i in range(1, 4)})

        semantic = self.semantic(features, mask_stride)
        instance = self.instance(features, mask_stride)
        geometry = self.geometry(features, mask_stride)

        pred = self.post_process(dict(**semantic, **instance, **geometry), mask)
        if target is not None:
            losses = self.loss(pred, target, frustum)
        else:
            losses = None

        return pred, losses

    def post_process(self, pred, mask):
        grid_dimensions = self.grid_dimensions
        semantic = pred['semantic3d']
        device = semantic.device
        stride = semantic.tensor_stride
        batch_size = semantic.C[:, 0].max().item() + 1
        dense_dimensions = torch.Size([batch_size, 1] + [gi // si for gi, si in zip(grid_dimensions, stride)])
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)

        for key in pred.keys():
            pred_item = pred[key]
            coord = pred_item.C.long()
            mask_item = mask[coord[:, 0], coord[:, 1], coord[:, 2], coord[:, 3]]
            pred_item = mask_invalid_sparse_voxels(pred_item, mask_item)
            if key == 'occupancy3d':
                default_value = -1.
            elif key == 'geometry':
                default_value = self.truncation
            else:
                default_value = 0.
            pred_item = dense_layer(pred_item, dense_dimensions, min_coordinates, default_value=default_value)[0]
            if stride[0] > 1:
                pred_item = F.interpolate(pred_item, size=grid_dimensions, mode='trilinear', align_corners=True)
            pred[key] = pred_item

        pred['weight_mask'] = mask[:, None]

        return pred

    def loss(self, pred, target, frustum):
        losses = dict()
        weight_mask = pred['weight_mask']
        occupancy_tg = (target['geometry'].abs() < 3.).float()[weight_mask]
        occupancy_pred = pred['occupancy3d'][weight_mask]
        loss_occupancy = self.loss_occupancy(occupancy_pred.sigmoid(), occupancy_tg)
        losses['Occupancy3D'] = loss_occupancy.mean() * self.loss_weight_occupancy
        weight_mask = (target['geometry'].abs() < 3.) * weight_mask
        weight = target['weighting3d'][weight_mask]

        semantic_pred = pred['semantic3d'].permute(0, 2, 3, 4, 1)[weight_mask[:, 0]]
        semantic_tg = target['semantic3d'][weight_mask[:, 0]]
        loss_semantic = self.loss_semantic(semantic_pred, semantic_tg)
        loss_semantic = loss_semantic * weight
        losses['Semantic3D'] = loss_semantic.mean() * self.loss_weight_semantic

        center_front = pred['weight_mask']  # * weight_mask
        center_pred = pred['center3d'][center_front]
        center_tg = target['instance3d']['center3d'][center_front]
        loss_center = self.loss_center(center_pred, center_tg)
        losses['Center3D'] = loss_center.sum() / (center_front.sum() + 1e-7) * self.loss_weight_center

        off_pred = pred['offset3d'].permute(0, 2, 3, 4, 1)[weight_mask[:, 0]]
        offset_tg = target['instance3d']['offset3d'].permute(0, 2, 3, 4, 1)[weight_mask[:, 0]]
        offset_front = (offset_tg.abs().sum(1) > 0)[:, None]
        loss_offset = self.loss_offset(off_pred, offset_tg) * offset_front
        losses['Offset3D'] = loss_offset.sum() / (offset_front.sum() + 1e-7) * self.loss_weight_offset

        geometry_pred = pred['geometry'][weight_mask]
        geometry_tg = target['geometry'].abs()[weight_mask]
        loss_geometry = self.loss_geometry(geometry_pred, geometry_tg)
        loss_geometry = loss_geometry * weight
        losses['Geometry'] = loss_geometry.mean() * self.loss_weight_geometry

        return losses
