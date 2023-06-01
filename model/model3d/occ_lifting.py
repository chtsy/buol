import torch
import torch.nn.functional as F
import MinkowskiEngine as Me
from torch import nn
from utils.back_projection import BackProjection
from utils.utils3d import mask_invalid_sparse_voxels

class OccupancyAwareLifting(nn.Module):
    def __init__(self, cfg):
        super(OccupancyAwareLifting, self).__init__()
        self.bp = BackProjection(cfg)

    def forward(self, pred, kept, mapping, data=None):
        semantic = pred['semantic2d'].argmax(1)
        features = pred['semantic2d'].softmax(1)
        depth = pred['depth'].detach().clone()
        depth_weight = pred['occupancy2d']
        depth_max_value = self.bp.depth_max
        batch = semantic.shape[0]

        depth[depth > depth_max_value] = depth_max_value

        depth_feat = (depth / depth_max_value * 100.)
        depth_index = depth_feat.long()
        depth_weight_kept = torch.ones_like(depth_weight, dtype=torch.long) * \
                            torch.arange(0, 100, device=depth.device, dtype=torch.long)[None, :, None, None]

        # stuff: wall, floor, or ceiling
        stuff = (-F.max_pool2d(-(semantic >= 10).float(), 5, 1, 2)).bool()
        stuff_depth = depth[:, 0] * stuff

        stuff_x_max = stuff_depth.max(1)[0]
        stuff_y_max = stuff_depth.max(2)[0]

        stuff_depth_l = stuff_depth[:, 0].clone()
        stuff_depth_r = stuff_depth[:, -1].clone()
        stuff_depth_t = stuff_depth[:, :, 0].clone()
        stuff_depth_d = stuff_depth[:, :, -1].clone()

        for bi in range(batch):
            stuff_depth[bi, 0] = stuff_padding(stuff_depth_l[bi], stuff_y_max[bi])
            stuff_depth[bi, -1] = stuff_padding(stuff_depth_r[bi], stuff_y_max[bi].flip(0))
            stuff_depth[bi, :, 0] = stuff_padding(stuff_depth_t[bi], stuff_x_max[bi])
            stuff_depth[bi, :, -1] = stuff_padding(stuff_depth_d[bi], stuff_x_max[bi].flip(0))

        stuff_x = stuff_depth.max(1)[0]
        stuff_y = stuff_depth.max(2)[0]

        for bi in range(batch):
            stuff_x[bi] = find_none(stuff_x[bi])
            stuff_y[bi] = find_none(stuff_y[bi])

        depth_pixels_xy = torch.ones_like(depth).nonzero()
        depth_max = torch.cat([
            stuff_x[depth_pixels_xy[:, 0], depth_pixels_xy[:, 3]][..., None],
            stuff_y[depth_pixels_xy[:, 0], depth_pixels_xy[:, 2]][..., None]],
            -1).min(-1)[0].reshape(*depth.shape)

        depth_max = (depth_max / depth_max_value * 100.).long()

        depth_feat = (depth_weight_kept - depth_index) / 100. * depth_max_value
        depth_feat = torch.cat([depth_feat.sign()[:, None], depth_feat[:, None].abs()], 1)

        depth_weight_kept = (depth_weight_kept > (depth_index - 3)) * (
                depth_weight_kept < (depth_max + 5))
        depth_weight = depth_weight.sigmoid() * depth_weight_kept
        feat_kept = kept.clone()

        if (data is not None) and ('room_mask' in data):
            room_mask = data['room_mask'][:, None]
            depth_weight_kept = depth_weight_kept * room_mask  # (targets['depth']['depth_map'] > 0)

            room_mask = F.interpolate(room_mask.float(), pred['center2d'].shape[-2:])
            pred['center2d'] = pred['center2d'] * room_mask

        mapping_kept = mapping[kept]
        mapping_kept[:, -1] = mapping_kept[:, -1] * 100 / 6
        mapping_kept = mapping_kept.long()

        feat_kept[kept] = depth_weight_kept[
            mapping_kept[:, 0], mapping_kept[:, -1], mapping_kept[:, 2], mapping_kept[:, 1]]
        features = torch.cat([features[:, :, None].repeat(1, 1, 100, 1, 1),
                              depth_weight[:, None], depth_feat], 1)

        coord_sparse = feat_kept.nonzero()
        mapping_feat_kept = mapping[feat_kept]

        mapping_feat_kept[:, -1] = mapping_feat_kept[:, -1] * 100 / depth_max_value
        mapping_feat_kept = mapping_feat_kept.long()
        feat_sparse = features[mapping_feat_kept[:, 0], :, mapping_feat_kept[:, -1],
                      mapping_feat_kept[:, 2], mapping_feat_kept[:, 1]]

        padding_kept = F.max_pool3d(feat_kept.float(), 5, 1, 2).bool()
        padding_kept[~kept] = False

        batch_point = padding_kept.flatten(1, -1).sum(-1)
        batch_zero = (batch_point == 0).nonzero().view(-1)
        # fix no points
        if len(batch_zero) > 0:
            padding_kept[batch_zero, 127, 127, 127] = True
        padding_kept[feat_kept] = False
        coord_padding = padding_kept.nonzero().contiguous().float()
        coord_padding[:, 1:] = coord_padding[:, 1:] // 2 * 2  ## decrease point
        feat_padding = torch.zeros((len(coord_padding), features.shape[1]), device=features.device, dtype=torch.float)

        feat_sparse = torch.cat([feat_sparse, feat_padding])
        coord_sparse = torch.cat([coord_sparse, coord_padding])
        proj_feat = Me.SparseTensor(
            features=feat_sparse,
            coordinates=coord_sparse.contiguous().float(),
            quantization_mode=Me.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)

        proj_feat = mask_invalid_sparse_voxels(proj_feat)
        coord = proj_feat.C.long()
        batch = coord[:, 0].max().item() + 1
        mask = torch.zeros((batch, * self.bp.frustum_dimensions), device=proj_feat.device, dtype=torch.bool, requires_grad=False)
        mask[coord[:, 0], coord[:, 1], coord[:, 2], coord[:, 3]] = True

        mask = F.max_pool3d(mask.float(), 5, 1, 2).bool()
        return proj_feat, mask


def stuff_padding(padding, max_value):
    padding = padding.clone()
    padding_mask = padding == 0
    if padding_mask.sum() > 0:
        for v in max_value:
            if v != 0:
                break
        padding[padding_mask] = v
    return padding


def find_none(stuff_a, min_value=0):
    none_v = torch.nonzero(stuff_a == 0)
    for v in none_v:
        l_stuff = stuff_a[:v]
        l_stuff = l_stuff[l_stuff != 0]
        l_stuff = min(l_stuff) if len(l_stuff) else min_value
        r_stuff = stuff_a[v + 1:]
        r_stuff = r_stuff[r_stuff != 0]
        r_stuff = min(r_stuff) if len(r_stuff) else min_value
        stuff_a[v] = max(l_stuff, r_stuff)
    return stuff_a
