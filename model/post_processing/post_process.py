import torch
from torch import nn
from .post_processing_2d import get_panoptic_segmentation
from .post_processing_3d import get_panoptic_segmentation_3d

class PostProcess(nn.Module):
    def __init__(self, cfg_post, cfg_proj):
        super(PostProcess, self).__init__()
        self.center_threshold = cfg_post.CENTER_THRESHOLD
        self.nms_kernel = cfg_post.NMS_KERNEL
        self.top_k = cfg_post.TOP_K_INSTANCE
        self.stuff_area = cfg_post.STUFF_AREA
        self.label_divisor = cfg_post.LABEL_DIVISOR
        self.thing_list = cfg_post.THING_LIST

        self.depth_min = cfg_proj.DEPTH_MIN
        self.depth_max = cfg_proj.DEPTH_MAX
        self.voxel_size = cfg_proj.VOXEL_SIZE
        self.truncation = cfg_proj.TRUNCATION

        self.is_mp = cfg_post.DATASET == "Matterport"

    def forward(self, pred):
        batch_size = pred['semantic2d'].shape[0]
        pred['geometry'][pred['occupancy3d'] <= 0] = self.truncation
        panoptic2d, center_point2d, classes2d, panoptic3d = [], [], [], []
        for bi in range(batch_size):
            panoptic, center_point, classes = get_panoptic_segmentation(
                pred['semantic2d'][bi:bi+1],
                pred['center2d'][bi:bi+1],
                pred['offset2d'][bi:bi+1],
                thing_list=self.thing_list,
                label_divisor=self.label_divisor,
                stuff_area=self.stuff_area,
                void_label=0,
                threshold=self.center_threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
                foreground_mask=None)

            panoptic2d.append(panoptic)
            center_point2d.append(center_point)
            classes2d.append(classes)

            panoptic = get_panoptic_segmentation_3d(
                pred["geometry"][bi, 0].abs() < (2. if self.is_mp else 1.5),
                pred["semantic3d"][bi].argmax(0),
                pred["offset3d"][bi],
                center_point, classes,
                pred['intrinsic'][bi],
                pred['semantic2d'].shape[-2:],
                thing_list=self.thing_list,
                label_divisor=self.label_divisor,
                stuff_area=self.stuff_area,
                depth_min=self.depth_min,
                depth_max=self.depth_max,
                voxel_size=self.voxel_size,)
            panoptic3d.append(panoptic)

        pred.update(dict(
            panoptic2d=torch.stack(panoptic2d),
            center_point2d=center_point2d,
            classes2d=classes2d,
            panoptic3d=torch.stack(panoptic3d)
        ))

        return pred


