from collections import OrderedDict

import torch
import torch.nn.functional as F
from utils.utils import compute_iou
from .panoptic_quality import PQStatCategory
from typing import Tuple, Dict
from utils.utils3d import thicken_grid


class PanopticReconstructionQuality(object):
    def __init__(self,
                 matching_threshold=0.25,
                 category_information=None,
                 ignore_labels=None,
                 reduction="mean",
                 dataset='',
                 label_divisor=1000):
        super().__init__()

        # Ignore freespace label and ceiling
        if ignore_labels is None:
            ignore_labels = [0,]# 12]
        self.ignore_labels = ignore_labels

        self.matching_threshold = matching_threshold
        # self.extract_mesh = extract_mesh

        if category_information is None:
            self.category_information = {
                -1: True,
                1: True,
                2: True,
                3: True,
                4: True,
                5: True,
                6: True,
                7: True,
                8: True,
                9: True,
                10: False,
                11: False,
                12: False ## 12 added for matterport
            }

        self.categories = {}

        for label, is_thing in self.category_information.items():
            self.categories[label] = PQStatCategory(is_thing)

        self.reduction = reduction

        self.is_mp = dataset == "Matterport"
        self.label_divisor = label_divisor

    def add(self, pred, target):
        batch = pred['semantic2d'].shape[0]
        distance_field_pred = pred["geometry"].cpu()
        frustum_mask = pred['frustum_mask'].cpu()
        for bi in range(batch):
            panoptic3d = pred['panoptic3d'][bi]
            panoptic3d = panoptic3d.squeeze().cpu()

            instance_semantic_classes_pred = {pano_id.item(): pano_id.item() // self.label_divisor for pano_id in
                                              panoptic3d[panoptic3d != 0].unique()}
            instance_information_pred = self._prepare_instance_masks(panoptic3d,
                                                                    instance_semantic_classes_pred,
                                                                    distance_field_pred[bi, 0],
                                                                    frustum_mask[bi])

            # Prepare GT masks
            distance_field_gt = target["geometry"][bi].squeeze().cpu()
            panoptic3d_gt = target["panoptic3d"][bi].squeeze().cpu()

            instance_semantic_classes_gt = {pano_id.item(): pano_id.item() // self.label_divisor for
                                            pano_id in panoptic3d_gt[panoptic3d_gt != 0].unique()}

            instance_information_gt = self._prepare_instance_masks(panoptic3d_gt, instance_semantic_classes_gt,
                                                                  distance_field_gt, frustum_mask[bi])
            self.add_single(instance_information_pred, instance_information_gt)


    def add_single(self, prediction: Dict[int, Tuple[torch.Tensor, int]], ground_truth: Dict[int, Tuple[torch.Tensor, int]]) -> None:
        matched_ids_ground_truth = set()
        matched_ids_prediction = set()

        per_sample_result = {}
        for label, is_thing in self.category_information.items():
            per_sample_result[label] = PQStatCategory(is_thing)

        # True Positives
        for ground_truth_instance_id, (ground_truth_instance_mask, ground_truth_semantic_label) in ground_truth.items():
            self.categories[ground_truth_semantic_label].n += 1
            per_sample_result[ground_truth_semantic_label].n += 1

            for prediction_instance_id, (prediction_instance_mask, prediction_semantic_label) in prediction.items():

                # 0: Check if prediction was already matched
                if prediction_instance_id in matched_ids_prediction:
                    continue

                # 1: Check if they have the same label
                are_same_category = ground_truth_semantic_label == prediction_semantic_label

                if not are_same_category:
                    continue

                # 2: Compute overlap and check if they are overlapping enough
                overlap = compute_iou(ground_truth_instance_mask, prediction_instance_mask)
                is_match = overlap > self.matching_threshold

                if is_match:
                    self.categories[ground_truth_semantic_label].iou += overlap
                    self.categories[ground_truth_semantic_label].tp += 1

                    per_sample_result[ground_truth_semantic_label].iou += overlap
                    per_sample_result[ground_truth_semantic_label].tp += 1

                    matched_ids_ground_truth.add(ground_truth_instance_id)
                    matched_ids_prediction.add(prediction_instance_id)
                    break

        # False Negatives
        for ground_truth_instance_id, (_, ground_truth_semantic_label) in ground_truth.items():

            # 0: Check if ground truth has not yet been matched
            if ground_truth_instance_id not in matched_ids_ground_truth:
                # self.logger.info(f"Not matched, counted as FN: {ground_truth_instance_id}, num voxels: {mask.sum()}")
                self.categories[ground_truth_semantic_label].fn += 1
                per_sample_result[ground_truth_semantic_label].fn += 1

        # False Positives
        for prediction_instance_id, (_, prediction_semantic_label) in prediction.items():

            # 0: Check if prediction has not yet been matched
            if prediction_instance_id not in matched_ids_prediction:
                # self.logger.info(f"Not matched, counted as FP: {prediction_instance_id}, num voxels: {mask.sum()}")
                self.categories[prediction_semantic_label].fp += 1
                per_sample_result[prediction_semantic_label].fp += 1

        return per_sample_result

    def reduce(self):
        if self.reduction == "mean":
            return self.reduce_mean()

        return None

    def reduce_mean(self):
        pq, sq, rq, n = 0, 0, 0, 0

        per_class_results = {}

        for class_label, class_stats in self.categories.items():
            iou = class_stats.iou
            tp = class_stats.tp
            fp = class_stats.fp
            fn = class_stats.fn
            num_samples = class_stats.n

            if tp + fp + fn == 0:
                per_class_results[class_label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'n': 0}
                continue

            if num_samples > 0:
                n += 1
                pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
                sq_class = iou / tp if tp != 0 else 0
                rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
                per_class_results[class_label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'n': num_samples}
                pq += pq_class
                sq += sq_class
                rq += rq_class

        results = OrderedDict()
        results.update({'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n})

        for label, per_class_result in per_class_results.items():
            if per_class_result["n"] > 0:
                results[f"pq_{label}"] = per_class_result["pq"]
                results[f"sq_{label}"] = per_class_result["sq"]
                results[f"rq_{label}"] = per_class_result["rq"]
                results[f"n_{label}"] = per_class_result["n"]

        return results

    def _prepare_instance_masks(self, instances, semantic_mapping, distance_field, frustum_mask):
        instance_information = {}

        if self.is_mp:
            frustum_mask = F.interpolate(frustum_mask.unsqueeze(0).unsqueeze(0).float(), scale_factor=0.5,
                                         mode="nearest", recompute_scale_factor=False).squeeze().bool()

        for instance_id, semantic_class in semantic_mapping.items():
            instance_mask: torch.Tensor = (instances == instance_id)
            instance_distance_field = torch.full_like(instance_mask, dtype=torch.float, fill_value=3.0)
            instance_distance_field[instance_mask] = distance_field.squeeze()[instance_mask]
            instance_distance_field_masked = instance_distance_field.abs() < (
                2.0 if self.is_mp else 1.)

            if self.is_mp:
                instance_distance_field_masked = F.max_pool3d(
                    instance_distance_field_masked.unsqueeze(0).unsqueeze(0).float(),
                    kernel_size=3, stride=2, padding=1).squeeze().bool()

                grid_dim = [128, 128, 128]
            else:
                grid_dim = [256, 256, 256]
            instance_grid = thicken_grid(instance_distance_field_masked, grid_dim, frustum_mask)
            instance_information[instance_id] = instance_grid, semantic_class

        return instance_information

