import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Union, List, Tuple
from torch.utils.data import Dataset
from PIL import Image
from utils import io
from utils import transforms2d as t2d
from utils import transforms3d as t3d
from utils.intrinsics import adjust_intrinsic
from utils.frustum import generate_frustum, generate_frustum_volume, compute_camera2frustum_transform

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class Matterport(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_root_path = Path(cfg.ROOT)
        self.samples = self.load_and_filter_file_list(cfg.FILE_LIST)

        # Fields defines which data should be loaded
        self.fields = cfg.FIELDS
        im_size = cfg.IMAGE_SIZE
        self.image_size = (im_size[1], im_size[0])
        dp_im_size = cfg.PROJECTION.IMAGE_SIZE
        self.depth_image_size = (dp_im_size[1], dp_im_size[0])
        self.voxel_size = cfg.PROJECTION.VOXEL_SIZE
        self.depth_min = cfg.PROJECTION.DEPTH_MIN
        self.depth_max = cfg.PROJECTION.DEPTH_MAX
        self.grid_dimensions = cfg.PROJECTION.GRID_DIMENSIONS
        self.truncation = cfg.PROJECTION.TRUNCATION
        self.stuff_classes = cfg.STUFF_CLASSES

        self.ignore_stuff_in_offset = cfg.IGNORE_STUFF_IN_OFFSET
        self.small_instance_weight = cfg.SMALL_INSTANCE_WEIGHT
        self.small_instance_area = cfg.SMALL_INSTANCE_AREA
        self.sigma = sigma = 8
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        z = x[:, np.newaxis, np.newaxis]
        x0, y0, z0 = 3 * sigma + 1, 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        self.g3d = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))

        self.transforms: Dict = self.define_transformations()

    def __getitem__(self, index) -> dict:
        sample_path = self.samples[index]
        scene_id = sample_path.split("/")[0]
        image_id = sample_path.split("/")[1]

        sample = dict()
        sample["sample_path"] = sample_path
        sample["index"] = index
        sample["name"] = sample_path

        # 2D data
        if "depth" in self.fields:
            # depth_path = self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "d", ".png")
            depth_path = self.dataset_root_path.parent / 'matterport_depth_gen' / scene_id / io.assemble_frame_name(image_id, "d", ".png")

            depth = Image.open(depth_path)
            depth = self.transforms["depth"](np.array(depth))

            # divide by depth unit
            depth.depth_map /= 4000

            # mask depth pixels outside the frustum (too close/too far)
            depth.depth_map[depth.depth_map > self.depth_max] = 0
            depth.depth_map[depth.depth_map < self.depth_min] = 0

            # load intrinsic
            intrinsic_path = self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "intrinsics_", ".npy", drop_yaw=True)
            intrinsic = np.load(intrinsic_path)#.reshape(3, 3)

            intrinsic = adjust_intrinsic(intrinsic, self.image_size, self.depth_image_size)
            intrinsic = torch.from_numpy(intrinsic).float()
            depth.intrinsic_matrix = intrinsic

            rm_path = self.dataset_root_path.parent / 'matterport_room_mask' / scene_id / io.assemble_frame_name(image_id, "rm", ".png")
            room_mask = np.array(Image.open(rm_path))
            sample["room_mask"] = self.transforms["room_mask"](room_mask) > 0

            sample["depth"] = {"depth_map": depth.depth_map}

        if "color" in self.fields:
            color = Image.open(self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "i", ".jpg"))#, formats=["JPEG"])
            color = self.transforms["color"](color)
            sample["image"] = {'color': color,
                               'intrinsic_matrix': depth.intrinsic_matrix}

            sample['size'] = np.array(self.image_size)

        if "instance2d" in self.fields:
            segmentation2d = np.load(self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "segmap", ".mapped.npz"))["data"]

            panoptic2d = self.process_panoptic(segmentation2d[:, :, :2])
            center_points = np.array(panoptic2d.pop('center_points'))
            instance_ids = panoptic2d.pop('instance_ids')
            class_ids = panoptic2d.pop('class_ids')
            instance2d = panoptic2d.pop('instance2d')

            if "panoptic" in self.fields:
                panoptic = self.process_panoptic3d(panoptic2d['semantic'], instance2d, instance_ids, class_ids)
                panoptic2d['panoptic'] = panoptic
            sample["panoptic2d"] = panoptic2d

        # 3D data
        needs_weighting = False
        if "geometry" in self.fields:
            geometry_path = self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "geometry", ".npz")
            geometry_content = np.load(geometry_path)
            geometry = self.transforms["geometry"](torch.from_numpy(geometry_content["data"]).flip(0, 1)).abs()

            mask = ~ self.transforms["mask"](torch.from_numpy(geometry_content["mask"]).flip(0, 1))

            # process hierarchy
            sample["occupancy3d"] = self.transforms["occupancy3d"](geometry)
            geometry = self.transforms["geometry_truncate"](geometry)
            sample["geometry"] = geometry

            # add frustum mask
            sample["frustum_mask"] = mask
            sample['image']["frustum_mask"] = mask

            needs_weighting = True

        if "semantic3d" in self.fields or "instance3d" in self.fields:
            segmentation3d_path = self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "segmentation", ".mapped.npz")
            semantic3d, instance3d = np.load(segmentation3d_path)["data"]
            semantic3d, instance3d = torch.from_numpy(semantic3d).flip(0, 1), torch.from_numpy(instance3d).flip(0, 1)
            needs_weighting = True

            if "semantic3d" in self.fields:
                semantic3d = self.transforms["semantic3d"](semantic3d)
                sample["semantic3d"] = semantic3d

            if "instance3d" in self.fields:
                instance3d = self.transforms["semantic3d"](instance3d)
                if "panoptic" in self.fields:
                    panoptic3d = self.process_panoptic3d(semantic3d, instance3d, instance_ids, class_ids)
                    sample['panoptic3d'] = panoptic3d
                instance3d = self.process_instance3d_2d(instance3d, geometry[0].abs() < 3., center_points, instance_ids, depth.intrinsic_matrix)
                sample["instance3d"] = instance3d

        if needs_weighting:
            weighting_path = self.dataset_root_path / scene_id / io.assemble_frame_name(image_id, "weighting", ".npz")
            weighting = torch.from_numpy(np.load(weighting_path, allow_pickle=True)["data"]).flip(0, 1)
            weighting = self.transforms["weighting3d"](weighting)
            sample["weighting3d"] = weighting

        return sample

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_and_filter_file_list(file_list_path: os.PathLike) -> List[str]:
        with open(file_list_path) as f:
            content = f.readlines()

        images = [line.strip() for line in content]

        return images

    def define_transformations(self) -> Dict:
        transforms = dict()

        # 2D transforms
        transforms["color"] = t2d.Compose([
            t2d.ToTensor(),
            # ColorJitter(0.4, 0.4, 0.4, 0.4),
            t2d.Normalize(_imagenet_stats["mean"], _imagenet_stats["std"])
        ])

        transforms["room_mask"] = t2d.Compose([
            t2d.ToImage(),
            t2d.Resize(self.depth_image_size, Image.NEAREST),
            t2d.ToNumpyArray(),
            t2d.ToTensorFromNumpy()
        ])

        transforms["depth"] = t2d.Compose([
            t2d.ToImage(),
            t2d.Resize(self.depth_image_size, Image.NEAREST),
            t2d.ToNumpyArray(),
            t2d.ToTensorFromNumpy(),
            t2d.ToDepthMap(None)   # Set image specific intrinsic later
        ])

        # 3D transforms
        transforms["geometry"] = t3d.Compose([
            t3d.ToTensor(dtype=torch.float),
            t3d.Unsqueeze(0),
            t3d.ToTSDF(truncation=12)
        ])

        transforms["mask"] = t3d.Compose([
            t3d.ToTensor(dtype=torch.bool),
        ])

        transforms["geometry_truncate"] = t3d.ToTSDF(truncation=self.truncation)
        transforms["occupancy3d"] = t3d.Compose([t3d.Absolute(),
                                                   t3d.ToBinaryMask(self.truncation),
                                                   t3d.ToTensor(dtype=torch.float)])
        transforms["weighting3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.float), t3d.Unsqueeze(0)])
        transforms["semantic3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.long)])
        transforms["instance3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.long), t3d.Mapping(mapping={}, ignore_values=[0])])

        return transforms


    def process_panoptic(self, target):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18
        Args:
            panoptic: numpy.array, colored image encoding panoptic label.
            segments: List, a list of dictionary containing information of every segment, it has fields:
                - id: panoptic id, after decoding `panoptic`.
                - category_id: semantic class id.
                - area: segment area.
                - bbox: segment bounding box.
                - iscrowd: crowd region.
        Returns:
            A dictionary with fields:
                - semantic: Tensor, semantic label, shape=(H, W).
                - foreground: Tensor, foreground mask label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(1, H, W).
                - center_points: List, center coordinates, with tuple (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is (offset_y, offset_x).
                - semantic_weights: Tensor, loss weight for semantic prediction, shape=(H, W).
                - center_weights: Tensor, ignore region of center prediction, shape=(H, W), used as weights for center
                    regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
                - offset_weights: Tensor, ignore region of offset prediction, shape=(H, W), used as weights for offset
                    regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
        """

        num_thing_class = 9
        num_class = 12
        semantic, panoptic = target[..., 0], target[..., 1]

        # panoptic = self.rgb2id(panoptic)
        height, width = panoptic.shape[0], panoptic.shape[1]
        # semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        foreground = (semantic > 0) & (semantic <= num_thing_class)# np.zeros_like(panoptic, dtype=np.uint8)
        instance2d = panoptic * foreground
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(panoptic, dtype=np.float32)
        x_coord = np.ones_like(panoptic, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        instance_ids = []
        class_ids = []
        for pano_id in np.unique(panoptic):
            cat_id = semantic[panoptic==pano_id][0]# seg["category_id"]
            #### fix
            if (cat_id > num_class) or (cat_id == 0):
                continue
            class_ids.append(cat_id)
            # if self.ignore_crowd_in_semantic:
            #     if not seg['iscrowd']:
            #         semantic[panoptic == seg["id"]] = cat_id
            # else:
            #     semantic[panoptic == seg["id"]] = cat_id
            # if cat_id in self.thing_list:
            #     foreground[panoptic == seg["id"]] = 1
            #if not seg['iscrowd']:
                # Ignored regions are not in `segments`.
                # Handle crowd region.
            center_weights[panoptic == pano_id] = 1
            if self.ignore_stuff_in_offset:
                # Handle stuff region.
                if (cat_id > 0) and (cat_id <= num_thing_class):
                    offset_weights[panoptic == pano_id] = 1
            else:
                offset_weights[panoptic == pano_id] = 1
            if (cat_id > 0) and (cat_id <= num_thing_class):
                # find instance center
                mask_index = np.where(panoptic == pano_id)
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic == pano_id] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
                center_pts.append([center_y, center_x])
                instance_ids.append(pano_id)

                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(
                    center[0, aa:bb, cc:dd], self.g[a:b, c:d])

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]


        return dict(
            semantic=torch.as_tensor(semantic.astype('long')),
            foreground=torch.as_tensor(foreground.astype('long')),
            center=torch.as_tensor(center.astype(np.float32)),
            center_points=center_pts,
            offset=torch.as_tensor(offset.astype(np.float32)),
            semantic_weights=torch.as_tensor(semantic_weights.astype(np.float32)),
            center_weights=torch.as_tensor(center_weights.astype(np.float32)),
            offset_weights=torch.as_tensor(offset_weights.astype(np.float32)),
            instance_ids=instance_ids,
            class_ids=class_ids,
            instance2d=torch.as_tensor(instance2d.astype(np.int64)),
        )

    def process_instance3d_2d(self, instance3d, occupancy, center_points, instance_ids, intrinsic):
        # Get GT intrinsic matrix
        shp = instance3d.shape
        camera2frustum = compute_camera2frustum_transform(
            intrinsic.cpu(), self.depth_image_size, self.depth_min, self.depth_max, self.voxel_size)

        intrinsic_inverse = torch.inverse(intrinsic)

        # projection
        offset3d = torch.zeros(3, *shp)
        center3d = torch.zeros(1, *shp)
        coordinates = torch.nonzero(occupancy)
        grid_coordinates = coordinates.clone()
        grid_coordinates[:, :2] = 256 - grid_coordinates[:, :2]

        padding_offsets = self.compute_frustum_padding(intrinsic_inverse)
        grid_coordinates = grid_coordinates - padding_offsets - torch.tensor([1., 1., 1.])
        grid_coordinates = torch.cat([grid_coordinates, torch.ones(len(grid_coordinates), 1)], 1)
        pointcloud = torch.mm(torch.inverse(camera2frustum), grid_coordinates.t())
        depth_pixels = torch.mm(intrinsic, pointcloud)

        yv = depth_pixels[0] / depth_pixels[2]
        xv = depth_pixels[1] / depth_pixels[2]
        zv = coordinates[:, 2].float()

        center_points = center_points / (self.image_size[0] / self.depth_image_size[0])

        for ci, ii in zip(center_points, instance_ids):
            mask_ii = (instance3d[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] == ii)
            if mask_ii.sum() == 0:
                continue
            z_mean = coordinates[mask_ii, 2].float().mean()

            offset_y_index = (ci[1] - yv) * mask_ii
            offset_x_index = (ci[0] - xv) * mask_ii
            offset_z_index = (z_mean - zv) * mask_ii
            offset3d[1, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] += offset_y_index
            offset3d[0, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] += offset_x_index
            offset3d[2, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] += offset_z_index

            distance = (xv - ci[0]) ** 2 + (yv - ci[1]) ** 2 + (coordinates[:, 2] - z_mean) ** 2
            center_id = np.argmin(distance)
            center = coordinates[center_id]

            # generate center heatmap
            width, height, depth = shp
            y, x, z = center
            sigma = self.sigma
            # upper left
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1)), int(np.round(z - 3 * sigma - 1))
            # bottom right
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2)), int(np.round(z + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], height) - ul[1]
            e, f = max(0, -ul[2]), min(br[2], depth) - ul[2]

            cc, dd = max(0, ul[0]), min(br[0], width)
            aa, bb = max(0, ul[1]), min(br[1], height)
            ee, ff = max(0, ul[2]), min(br[2], depth)
            center3d[0, aa:bb, cc:dd, ee:ff] = np.maximum(
                center3d[0, aa:bb, cc:dd, ee:ff], self.g3d[a:b, c:d, e:f])

        return dict(
            offset3d=offset3d,
            center3d=center3d,
            weight3d=occupancy[None],
        )

    def process_instance3d(self, semantic, instance, instance_ids, intrinsic):
        num_thing_class = 9

        # panoptic = self.rgb2id(panoptic)
        height, width, depth = instance.shape[0], instance.shape[1], instance.shape[2],
        # semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
        foreground = (semantic > 0) & (semantic <= num_thing_class)# np.zeros_like(panoptic, dtype=np.uint8)
        center = np.zeros((1, height, width, depth), dtype=np.float32)
        center_pts = []
        offset = np.zeros((3, height, width, depth), dtype=np.float32)
        y_coord = np.ones_like(instance, dtype=np.float32)
        x_coord = np.ones_like(instance, dtype=np.float32)
        z_coord = np.ones_like(instance, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        z_coord = np.cumsum(z_coord, axis=2) - 1

        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(instance, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_label`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(instance, dtype=np.uint8)
        offset_weights = np.zeros_like(instance, dtype=np.uint8)
        for pano_id in instance_ids:
            if pano_id == 0:
                continue
            cat_id = semantic[instance==pano_id][0]
            center_weights[instance == pano_id] = 1
            if self.ignore_stuff_in_offset:
                # Handle stuff region.
                if (cat_id > 0) and (cat_id <= num_thing_class):
                    offset_weights[instance == pano_id] = 1
            else:
                offset_weights[instance == pano_id] = 1
            if (cat_id > 0) and (cat_id <= num_thing_class):
                # find instance center
                mask_index = np.where(instance == pano_id)
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[instance == pano_id] = self.small_instance_weight

                center_y, center_x, center_z = np.mean(mask_index[0]), np.mean(mask_index[1]), np.mean(mask_index[2])
                center_pts.append([center_y, center_x, center_z])

                # generate center heatmap
                y, x, z = int(center_y), int(center_x), int(center_z)
                # outside image boundary
                if x < 0 or y < 0 or z < 0 or \
                        x >= width or y >= height or z >= depth:
                    continue
                sigma = self.sigma
                # upper left
                ulf = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1)), int(np.round(z - 3 * sigma - 1))
                # bottom right
                brb = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2)), int(np.round(z + 3 * sigma + 2))

                c, d = max(0, -ulf[0]), min(brb[0], width) - ulf[0]
                a, b = max(0, -ulf[1]), min(brb[1], height) - ulf[1]
                e, f = max(0, -ulf[2]), min(brb[2], depth) - ulf[2]

                cc, dd = max(0, ulf[0]), min(brb[0], width)
                aa, bb = max(0, ulf[1]), min(brb[1], height)
                ee, ff = max(0, ulf[2]), min(brb[2], depth)
                center[0, aa:bb, cc:dd, ee:ff] = np.maximum(
                    center[0, aa:bb, cc:dd, ee:ff], self.g3d[a:b, c:d, e:f])

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1], mask_index[2])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1], mask_index[2])
                offset_d_index = (2*np.ones_like(mask_index[0]), mask_index[0], mask_index[1], mask_index[2])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]
                offset[offset_d_index] = center_z - z_coord[mask_index]

        return dict(
            offset3d=torch.as_tensor(offset.astype(np.float32)),
            center3d=torch.as_tensor(center.astype(np.float32))
        )


    def compute_frustum_padding(self, intrinsic_inverse: torch.Tensor) -> torch.Tensor:
        depth_size = self.depth_image_size
        depth_size = [depth_size[1], depth_size[0]]
        frustum = generate_frustum(depth_size, intrinsic_inverse.cpu(), self.depth_min, self.depth_max)
        dimensions, _ = generate_frustum_volume(frustum, self.voxel_size)
        difference = (torch.tensor([256, 256, 256]) - torch.tensor(dimensions)).float().to(intrinsic_inverse.device)

        padding_offsets = difference // 2

        return padding_offsets

    def process_panoptic3d(self, sem, ins, instance_ids, class_ids):
        thing_list = range(1, 10)
        label_divisor = 1000
        class_id_tracker = {}
        pano = torch.zeros_like(sem)
        for ins_id in instance_ids:
            if ins_id==0:
                continue
            thing_mask = (ins == ins_id)
            if thing_mask.sum()==0:
                continue
            class_id = sem[thing_mask][0]
            if class_id.item() in class_id_tracker:
                new_ins_id = class_id_tracker[class_id.item()]
            else:
                class_id_tracker[class_id.item()] = 1
                new_ins_id = 1
            class_id_tracker[class_id.item()] += 1
            pano[thing_mask] = class_id * label_divisor + new_ins_id

        # paste stuff to unoccupied area
        for class_id in class_ids:
            if class_id.item() in thing_list:
                # thing class
                continue
            # calculate stuff area
            stuff_mask = (sem == class_id)
            pano[stuff_mask] = class_id * label_divisor

        return pano






