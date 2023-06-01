import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_loss_info_str(loss_meter_dict):
    msg = ''
    for key in loss_meter_dict.keys():
        msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            name=key, meter=loss_meter_dict[key]
        )

    return msg

def intersection(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return (ground_truth.bool() & prediction.bool()).float()


def union(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return (ground_truth.bool() | prediction.bool()).float()


def compute_iou(ground_truth: torch.Tensor, prediction: torch.Tensor) -> float:
    num_intersection = float(torch.sum(intersection(ground_truth, prediction)))
    num_union = float(torch.sum(union(ground_truth, prediction)))
    iou = 0.0 if num_union == 0 else num_intersection / num_union
    return iou

def to_device(data, device):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
        elif isinstance(data[key], dict):
            for sub_key in data[key]:
                    data[key][sub_key] = data[key][sub_key].to(device)
    return data