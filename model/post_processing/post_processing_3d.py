# ------------------------------------------------------------------------------
# Post-processing to get instance and panoptic segmentation results.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# Modified by Tao Chu
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F

from utils.frustum import generate_frustum, generate_frustum_volume, compute_camera2frustum_transform


def get_semantic_segmentation(sem):
    """
    Post-processing for semantic segmentation branch.
    Arguments:
        sem: A Tensor of shape [N, C, H, W], where N is the batch size, for consistent, we only
            support N=1.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    sem = sem.squeeze(0)
    return torch.argmax(sem, dim=0, keepdim=True)


def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1])


def group_pixels(ctr, offsets):
    """
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    """
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)

    ctr_loc = coord + offsets
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    # distance: [K, H*W]
    distance = torch.norm(ctr - ctr_loc, dim=-1)

    # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
    return instance_id


def get_instance_segmentation(sem_seg, ctr_hmp, offsets, thing_list, threshold=0.1, nms_kernel=3, top_k=None,
                              thing_seg=None):
    """
    Post-processing for instance segmentation, gets class agnostic instance id map.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list: A List of thing class id.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
        thing_seg: A Tensor of shape [1, H, W], predicted foreground mask, if not provided, inference from
            semantic prediction.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
        A Tensor of shape [1, K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if thing_seg is None:
        # gets foreground segmentation
        thing_seg = torch.zeros_like(sem_seg)
        for thing_class in thing_list:
            thing_seg[sem_seg == thing_class] = 1

    ctr = find_instance_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)
    if ctr.size(0) == 0:
        return torch.zeros_like(sem_seg), ctr.unsqueeze(0)
    ins_seg = group_pixels(ctr, offsets)
    return thing_seg * ins_seg, ctr.unsqueeze(0)


def merge_semantic_and_instance(sem_seg, ins_seg, panoptic2d, label_divisor, thing_list, stuff_area, void_label):
    """
    Post-processing for panoptic segmentation, by merging semantic segmentation label and class agnostic
        instance segmentation label.
    Arguments:
        sem_seg: A Tensor of shape [1, H, W], predicted semantic label.
        ins_seg: A Tensor of shape [1, H, W], predicted instance label.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        thing_list: A List of thing class id.
        stuff_area: An Integer, remove stuff whose area is less tan stuff_area.
        void_label: An Integer, indicates the region has no confident prediction.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    Raises:
        ValueError, if batch size is not 1.
    """
    # In case thing mask does not align with semantic prediction
    pan_seg = torch.zeros_like(sem_seg) + void_label
    thing_seg = ins_seg > 0
    semantic_thing_seg = torch.zeros_like(sem_seg)
    for thing_class in thing_list:
        semantic_thing_seg[sem_seg == thing_class] = 1

    # keep track of instance id for each class
    class_id_tracker = {}

    semantic_ids = sem_seg.unique()
    for sem_id in semantic_ids:
        if sem_id not in thing_list:
            continue
        mask = (sem_seg == sem_id)
        if sem_id not in panoptic2d['class2d'].values():
            sem_seg[mask] = 0
            continue
        for ins_id in panoptic2d['class2d']:
            pass

        instance_ids = ins_seg[mask].unique()
        for ins_id in instance_ids:
            print(sem_id, ins_id, (mask * (ins_seg==ins_id)).sum())


    # paste thing by majority voting
    instance_ids = torch.unique(ins_seg)
    for ins_id in instance_ids:
        if ins_id == 0:
            continue
        # Make sure only do majority voting within semantic_thing_seg
        thing_mask = (ins_seg == ins_id) & (semantic_thing_seg == 1)
        if torch.nonzero(thing_mask).size(0) == 0:
            continue
        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1, ))
        if class_id.item() in class_id_tracker:
            new_ins_id = class_id_tracker[class_id.item()]
        else:
            class_id_tracker[class_id.item()] = 1
            new_ins_id = 1
        class_id_tracker[class_id.item()] += 1
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id

    # paste stuff to unoccupied area
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_list:
            # thing class
            continue
        # calculate stuff area
        stuff_mask = (sem_seg == class_id) & (~thing_seg)
        area = torch.nonzero(stuff_mask).size(0)
        if area >= stuff_area:
            pan_seg[stuff_mask] = class_id * label_divisor

    return pan_seg


def get_panoptic_segmentation_3d(geometry, semantic3d, offset3d, center_point2d, classes2d,
                                 intrinsic, image_size, thing_list, label_divisor,
                                 stuff_area, depth_min, depth_max, voxel_size):
    panoptic3d = torch.zeros_like(semantic3d)
    ctr = center_point2d.permute(1, 0, 2)# // 2
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

    ##### filter
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

def max_pooling_by_panoptic(pano, kernel_size=3):
    for pu in pano.unique():
        if pu != 0:
            # kernel = 7 if (kernel == 5) and (pu==10000)
            pano_mask = (pano == pu)
            not_mask = (pano == 0) + (~pano_mask)
            pano_mask = F.max_pool3d(
                pano_mask[None].float(),
                kernel_size=kernel_size,
                stride=1, padding=kernel_size // 2).bool()[0]
            pano_mask = pano_mask * not_mask
            pano[pano_mask] = pu
    return pano


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


def nn_search_old(grid, point, radius=3):
    start = -radius
    end = radius
    labels = []
    for x in range(start, end):
        for y in range(start, end):
            for z in range(start, end):
                offset = torch.tensor([x, y, z], device=point.device)
                point_offset = point + offset
                label_bi = grid[point_offset[:, 0],
                                point_offset[:, 1],
                                point_offset[:, 2]]

                if label_bi.sum() != 0:
                    labels.append(label_bi[:, None])

    labels = torch.cat(labels, 1)
    set_labels = torch.zeros([len(labels)], dtype=labels.dtype, device=labels.device)
    for i, li in enumerate(labels):
        li_u, count_li = li.unique(return_counts=True)
        max_count = 0
        max_label = -1
        for lii, cii in zip(li_u, count_li):
            if lii != 0:
                if cii > max_count:
                    max_count = cii
                    max_label = lii
        set_labels[i] = max_label

    return set_labels
