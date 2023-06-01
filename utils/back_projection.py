import torch
import MinkowskiEngine as Me
from torch import nn
from .frustum import generate_frustum, generate_frustum_volume, compute_camera2frustum_transform


class BackProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.IMAGE_SIZE
        self.depth_min = cfg.DEPTH_MIN
        self.depth_max = cfg.DEPTH_MAX
        self.voxel_size = cfg.VOXEL_SIZE
        self.frustum_dimensions = torch.tensor(cfg.GRID_DIMENSIONS)

    def forward(self, shp, intrinsics, frustum_masks=None, room_masks=None) -> Me.SparseTensor:
        device = intrinsics.device
        if frustum_masks is None:
            frustum_masks = torch.ones([len(intrinsics), *self.frustum_dimensions], dtype=torch.bool, device=device)
        len_shp = len(frustum_masks.shape)
        if len_shp == 3:
            frustum_masks = frustum_masks[None]
            intrinsics = intrinsics[None]

        kepts, mappings = [], []
        for bi, (intrinsic, frustum_mask) in enumerate(zip(intrinsics, frustum_masks)):
            camera2frustum = compute_camera2frustum_transform(intrinsic.cpu(), self.image_size, self.depth_min,
                                                              self.depth_max, self.voxel_size).cuda(device)
            intrinsic_inverse = torch.inverse(intrinsic)
            coordinates = torch.nonzero(frustum_mask)
            grid_coordinates = coordinates.clone()
            grid_coordinates[:, :2] = 256 - grid_coordinates[:, :2]

            padding_offsets = self.compute_frustum_padding(intrinsic_inverse)
            grid_coordinates = grid_coordinates - padding_offsets - torch.tensor([1., 1., 1.], device=device)
            grid_coordinates = torch.cat([grid_coordinates, torch.ones(len(grid_coordinates), 1, device=device)],
                                         1)
            pointcloud = torch.mm(torch.inverse(camera2frustum), grid_coordinates.t())
            depth_pixels = torch.mm(intrinsic, pointcloud)

            depth = depth_pixels[2]
            coord = depth_pixels[0:2] / depth
            coord = torch.cat([coord, coordinates[:, 2][None]], 0).permute(1, 0)

            kept = (depth <= self.depth_max) * \
                   (depth >= self.depth_min) * \
                   (coord[:, 0] < shp[1]) * (coord[:, 1] < shp[0])
            coordinates = coordinates[kept]
            depth = depth[kept, None]

            mapping = torch.zeros(256, 256, 256, 5, device=depth.device) - 1.
            mapping[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] =\
                torch.cat([torch.ones_like(depth) * bi, coord[kept], depth], -1)

            kept = (mapping >= 0).all(-1)

            if room_masks is not None:
                mapping_kept = mapping[kept].long()
                kept[kept.clone()] = room_masks[bi, 0, mapping_kept[:, 2], mapping_kept[:, 1]]

            kepts.append(kept)
            mappings.append(mapping)

        if len_shp == 3:
            kepts = kepts[0]
            mappings = mappings[0, ..., 1:]
        else:
            kepts = torch.stack(kepts, 0)
            mappings = torch.stack(mappings, 0)

        return kepts, mappings

    def compute_frustum_padding(self, intrinsic_inverse: torch.Tensor) -> torch.Tensor:
        frustum = generate_frustum(self.image_size, intrinsic_inverse.cpu(), self.depth_min, self.depth_max)
        dimensions, _ = generate_frustum_volume(frustum, self.voxel_size)
        difference = (self.frustum_dimensions - torch.tensor(dimensions)).float().to(intrinsic_inverse.device)

        padding_offsets = difference / 2
        return padding_offsets
