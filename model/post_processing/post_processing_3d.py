import torch
from utils.frustum import generate_frustum, generate_frustum_volume, compute_camera2frustum_transform


def get_panoptic_segmentation_3d(geometry, semantic3d, offset3d, center_point2d, classes2d,
                                 intrinsic, image_size, thing_list, label_divisor,
                                 stuff_area, depth_min, depth_max, voxel_size):
    panoptic3d = torch.zeros_like(semantic3d)
    ctr = center_point2d.permute(1, 0, 2)
    ctr = ctr.flip(2)
    semantic3d = semantic3d * geometry
    thing_seg = torch.zeros_like(semantic3d)

    for thing_class in thing_list:
        thing_seg[semantic3d == thing_class] = 1
    thing_seg = thing_seg.bool()

    # Get GT intrinsic matrix
    camera2frustum = compute_camera2frustum_transform(intrinsic.cpu(), [image_size[1], image_size[0]], depth_min,
                                                      depth_max, voxel_size).cuda(geometry.device)
    intrinsic_inverse = torch.inverse(intrinsic)

    # projection
    coordinates = torch.nonzero(geometry)
    grid_coordinates = coordinates.clone()
    grid_coordinates[:, :2] = 256 - grid_coordinates[:, :2]
    padding_offsets = compute_frustum_padding(intrinsic_inverse, image_size, depth_min, depth_max, voxel_size)
    grid_coordinates = grid_coordinates - padding_offsets - torch.tensor([1., 1., 1.], device=geometry.device)
    grid_coordinates = torch.cat([grid_coordinates, torch.ones(len(grid_coordinates), 1, device=geometry.device)], 1)
    pointcloud = torch.mm(torch.inverse(camera2frustum), grid_coordinates.t())
    depth_pixels = torch.mm(intrinsic, pointcloud)

    depth = depth_pixels[2]
    coord = depth_pixels[0:2] / depth

    coord_offset = offset3d[:2, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
    coord_offset = coord_offset.flip(0)

    pred_center = (coord + coord_offset)[None].permute(0, 2, 1)

    for sem_id in semantic3d.unique():
        if sem_id not in thing_list:
            continue
        mask = (semantic3d == sem_id)
        if sem_id not in classes2d:
            semantic3d[mask] = 0
            continue
        ins_ids = torch.nonzero(classes2d==sem_id).view(-1)
        if len(ins_ids) == 1:
            panoptic3d[mask] = sem_id * label_divisor + 1
        else:
            ctr_ci = ctr[ins_ids]
            distance = torch.norm(ctr_ci - pred_center, dim=-1)
            instance_id = torch.argmin(distance, dim=0) + 1
            instance_id = instance_id[mask[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]]
            coord_mask = coordinates[mask[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]]
            panoptic3d[coord_mask[:, 0], coord_mask[:, 1], coord_mask[:, 2]] = sem_id * label_divisor + instance_id

    class_ids = torch.unique(semantic3d)
    for class_id in class_ids:
        if class_id.item() in thing_list:
            # thing class
            continue
        # calculate stuff area
        stuff_mask = (semantic3d == class_id) & (~thing_seg)
        area = torch.nonzero(stuff_mask).size(0)
        if area >= stuff_area:
            panoptic3d[stuff_mask] = class_id * label_divisor

    pano_ids = torch.unique(panoptic3d)
    for pano_id in pano_ids:
        # calculate stuff area
        pano_mask = panoptic3d == pano_id
        if pano_mask.sum() < 100:
            panoptic3d[pano_mask] = 0

    unassigned_voxels = (geometry & (panoptic3d == 0).bool()).nonzero()

    if len(unassigned_voxels) > 0:
        panoptic3d_copy = panoptic3d.clone()
        panoptic3d[
            unassigned_voxels[:, 0],
            unassigned_voxels[:, 1],
            unassigned_voxels[:, 2]] = nn_search(panoptic3d_copy, unassigned_voxels)

    return panoptic3d[None]


def compute_frustum_padding(intrinsic_inverse: torch.Tensor,
                            image_size, depth_min, depth_max, voxel_size) -> torch.Tensor:
    frustum = generate_frustum(image_size, intrinsic_inverse.cpu(), depth_min, depth_max)
    dimensions, _ = generate_frustum_volume(frustum, voxel_size)
    difference = (torch.tensor([256, 256, 256]) - torch.tensor(dimensions)).float().to(intrinsic_inverse.device)

    padding_offsets = difference // 2

    return padding_offsets

def nn_search(grid, point, radius=3):
    start = -radius
    end = radius
    label = torch.zeros([len(point)], device=point.device, dtype=grid.dtype)
    mask = torch.zeros_like(label).bool()

    start_to_end = torch.arange(start, end)
    start_to_end = start_to_end[start_to_end.abs().argsort()]

    for x in start_to_end:
        for y in start_to_end:
            for z in start_to_end:
                offset = torch.tensor([x, y, z], device=point.device)
                point_offset = point + offset
                label_bi = grid[point_offset[:, 0],
                                point_offset[:, 1],
                                point_offset[:, 2]]

                if label_bi.sum() != 0:
                    new_mask = (label_bi > 0) * (~mask)
                    label[new_mask] = label_bi[new_mask]
                    mask = mask + new_mask
    return label

