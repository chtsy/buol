import torch
import torch.nn.functional as F
from torch import nn
from .model2d.resnet import backbone as bb2d
from .model2d.depth import DepthPredictor
from .model2d.bottom_up import BottomUp2D
from .model3d.resnet_sparse import backbone as bb3d
from .model3d.bottom_up import BottomUp3D
from .model3d.occ_lifting import OccupancyAwareLifting
from .post_processing.post_process import PostProcess


class BUOL(nn.Module):
    def __init__(self, cfg):
        super(BUOL, self).__init__()
        self.bb2d = bb2d(cfg.MODEL2D.BACKBONE)
        self.bu2d = BottomUp2D(cfg.MODEL2D.BOTTOM_UP)
        self.depth = DepthPredictor(cfg.MODEL2D.DEPTH)
        self.bb3d = bb3d(cfg.MODEL3D.BACKBONE)
        self.ol = OccupancyAwareLifting(cfg.PROJECTION)
        self.bu3d = BottomUp3D(cfg.MODEL3D.BOTTOM_UP)
        self.post_process = PostProcess(cfg.POST_PROCESSING, cfg.PROJECTION)

    def forward(self, image, data=None):
        feat2d = self.bb2d(image['color'])
        pred2d, loss2d = self.bu2d(feat2d, data)

        with torch.no_grad():
            if 'frustum_mask' in image:
                frustum = image['frustum_mask']
                kept, mapping = self.ol.bp(self.bu2d.im_size + (256,), image['intrinsic_matrix'], frustum)
            else:
                kept, mapping = self.ol.bp(self.bu2d.im_size + (256,), image['intrinsic_matrix'], None)
                frustum = kept

        if data is not None:
            data = self.gen_occ2d_tg(data, kept, mapping)

        depth, loss_dp = self.depth(feat2d, data)
        pred2d.update(depth)

        feat3d, mask3d = self.ol(pred2d, kept, mapping, data)
        feat3d = self.bb3d(feat3d)
        pred3d, loss3d = self.bu3d(feat3d, mask3d, frustum, data)

        pred = dict(**pred2d, **pred3d)
        if not self.training:
            pred['frustum_mask'] = kept
            pred['intrinsic'] = image['intrinsic_matrix']
            pred = self.post_process(pred)

        if data is not None:
            loss = dict(**loss2d, **loss_dp, **loss3d)
            return pred, loss
        else:
            return pred

    def gen_occ2d_tg(self, data, kept, mapping):
        batch_size = kept.shape[0]
        mapping_kept = mapping[kept]
        mapping_kept[:, -1] = mapping_kept[:, -1] * 100 / 6
        mapping_kept = mapping_kept.long()
        device = kept.device

        depth_occ_tg = torch.zeros(
            (batch_size, self.depth.prediction['occupancy2d'])
            + self.bu2d.im_size, device=device)

        occupancy = data['occupancy3d'][:, 0]
        depth_occ_tg[mapping_kept[:, 0], mapping_kept[:, -1], mapping_kept[:, 2], mapping_kept[:, 1]] = occupancy[
            kept].float()

        depth_occ_tg = F.max_pool3d(depth_occ_tg, 3, 1, 1)
        data['depth']['occupancy'] = depth_occ_tg

        depth_tg = data['depth']['depth_map'][:, None]
        depth_index = (depth_tg / 6. * 100.).long()
        depth_weight_kept = torch.ones_like(depth_occ_tg) * \
                            torch.arange(0, 100, device=device)[None, :, None, None]
        depth_weight_kept = (depth_weight_kept > depth_index) * (
                depth_weight_kept < (depth_index.flatten(1, 3).max(-1)[0] + 10)[:, None, None, None])

        depth_weight_kept = F.max_pool3d(depth_weight_kept.float(), 5, 1, 2).bool()

        if 'room_mask' in data:
            room_mask = data['room_mask'][:, None]
            depth_weight_kept = depth_weight_kept * room_mask

        data['depth']['occupancy_weight'] = depth_weight_kept

        return data
