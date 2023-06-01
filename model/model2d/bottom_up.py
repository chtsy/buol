from torch import nn
import torch.nn.functional as F
from .predictor import Predictor
from ..loss import loss


class BottomUp2D(nn.Module):
    def __init__(self, cfg):
        super(BottomUp2D, self).__init__()
        self.semantic = Predictor(cfg.IN_FEATURE, cfg.SEMANTIC)
        self.instance = Predictor(cfg.IN_FEATURE, cfg.INSTANCE)
        self.im_size = cfg.IMAGE_SIZE

        self.loss_semantic = loss(cfg.LOSS.SEMANTIC)
        self.loss_center = loss(cfg.LOSS.CENTER)
        self.loss_offset = loss(cfg.LOSS.OFFSET)

        self.loss_weight_semantic = cfg.LOSS.SEMANTIC.WEIGHT
        self.loss_weight_center = cfg.LOSS.CENTER.WEIGHT
        self.loss_weight_offset = cfg.LOSS.OFFSET.WEIGHT

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
            Args:
                pred (dict): stores all output of the segmentation model.
                input_shape (tuple): spatial resolution of the desired shape.
            Returns:
                result (OrderedDict): upsampled dictionary.
            """
        # Override upsample method to correctly handle `offset`
        result = dict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def forward(self, features, target=None):
        semantic = self.semantic(features)
        instance = self.instance(features)

        pred = dict(**semantic, **instance)
        if target is not None:
            losses = self.loss(pred, target['panoptic2d'])
        else:
            losses = None

        pred = self._upsample_predictions(pred, self.im_size)
        return pred, losses

    def loss(self, pred, target):
        eps = 1e-7
        im_size = target['semantic'].shape[-2:]
        pred = self._upsample_predictions(pred, im_size)

        losses = dict()


        loss_weights_semantic = target['semantic_weights']
        loss_semantic = self.loss_semantic(
            pred['semantic2d'], target['semantic']) * loss_weights_semantic
        losses['Semantic2D'] = loss_semantic.sum() / (
                loss_weights_semantic.sum() + eps) * self.loss_weight_semantic

        loss_weights_center = target['center_weights'][:, None, :, :]
        loss_center = self.loss_center(
            pred['center2d'], target['center']) * loss_weights_center
        losses['Center2D'] = loss_center.sum() / (
                loss_weights_center.sum() + eps) * self.loss_weight_center

        loss_weights_offset = target['offset_weights'][:, None, :, :].expand_as(pred['offset2d'])
        loss_offset = self.loss_offset(pred['offset2d'], target['offset']) * loss_weights_offset
        losses['Offset2D'] = loss_offset.sum() / (
                loss_weights_offset.sum() + eps) * self.loss_weight_offset

        return losses


