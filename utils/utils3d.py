from typing import Tuple, Dict

import MinkowskiEngine as Me
import torch

# Type hints
ModuleResult = Tuple[Dict, Dict]


def sparse_cat_union(a: Me.SparseTensor, b: Me.SparseTensor):
    cm = a.coordinate_manager
    assert cm == b.coordinate_manager, "different coords_man"
    assert a.tensor_stride == b.tensor_stride, "different tensor_stride"

    zeros_cat_with_a = torch.zeros([a.F.shape[0], b.F.shape[1]], dtype=a.dtype).to(a.device)
    zeros_cat_with_b = torch.zeros([b.F.shape[0], a.F.shape[1]], dtype=a.dtype).to(a.device)

    feats_a = torch.cat([a.F, zeros_cat_with_a], dim=1)
    feats_b = torch.cat([zeros_cat_with_b, b.F], dim=1)

    new_a = Me.SparseTensor(
        features=feats_a,
        coordinates=a.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    new_b = Me.SparseTensor(
        features=feats_b,
        coordinates=b.C,
        coordinate_manager=cm,
        tensor_stride=a.tensor_stride,
    )

    return new_a + new_b


def get_sparse_values(tensor: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    values = tensor[coordinates[:, 0], :, coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]]
    return values

def thicken_grid(grid, grid_dims, frustum_mask):
    offsets = torch.nonzero(torch.ones(3, 3, 3)).long().to(grid.device)
    locs_grid = grid.nonzero(as_tuple=False)
    locs = locs_grid.unsqueeze(1).repeat(1, 27, 1)
    locs += offsets
    locs = locs.view(-1, 3)
    mask_x = (locs[:, 0] >= 0) & (locs[:, 0] < grid_dims[0])
    mask_y = (locs[:, 1] >= 0) & (locs[:, 1] < grid_dims[1])
    mask_z = (locs[:, 2] >= 0) & (locs[:, 2] < grid_dims[2])
    locs = locs[mask_x & mask_y & mask_z]

    thicken = torch.zeros(grid_dims, dtype=torch.bool, device=grid.device)
    thicken[locs[:, 0], locs[:, 1], locs[:, 2]] = True
    # frustum culling
    thicken = thicken & frustum_mask

    return thicken


def mask_invalid_sparse_voxels(grid: Me.SparseTensor, mask=None, frustum_dim=[256, 256, 256]) -> Me.SparseTensor:
    # Mask out voxels which are outside of the grid
    valid_mask = (grid.C[:, 1] < frustum_dim[0] - 1) & (grid.C[:, 1] >= 0) & \
                 (grid.C[:, 2] < frustum_dim[1] - 1) & (grid.C[:, 2] >= 0) & \
                 (grid.C[:, 3] < frustum_dim[2] - 1) & (grid.C[:, 3] >= 0)
    if mask is not None:
        valid_mask = valid_mask * mask
    num_valid_coordinates = valid_mask.sum()

    if num_valid_coordinates == 0:
        return {}, {}

    num_masked_voxels = grid.C.size(0) - num_valid_coordinates
    grids_needs_to_be_pruned = num_masked_voxels > 0

    # Fix: Only prune if there are invalid voxels
    if grids_needs_to_be_pruned:
        grid = Me.MinkowskiPruning()(grid, valid_mask)

    return grid

