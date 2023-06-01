# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.utils.data
import numpy as np
import random
from PIL import Image
from torchvision import transforms as T
from typing import Dict, Optional, List, Tuple

from utils.depth_map import DepthMap


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class CenterCrop:
    def __init__(self, size_image):
        self.image_size = size_image

    def __call__(self, image):
        image = self.center_crop(image)

        return image

    def center_crop(self, image):
        w1, h1 = image.size
        tw, th = self.image_size

        if w1 == tw and h1 == th:
            return image

        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        image = image.crop((x1, y1, tw + x1, th + y1))

        return image


class FromDistanceToDepth:
    def __init__(self, focal_length: float) -> None:
        self.focal_length = focal_length

    def __call__(self, distance_image: np.array) -> np.array:
        width = distance_image.shape[0]
        height = distance_image.shape[1]

        cx = width // 2
        cy = height // 2

        xs = np.arange(width) - cx
        ys = np.arange(height) - cy
        xis, yis = np.meshgrid(ys, xs)

        depth = np.sqrt(
            distance_image ** 2 / ((xis ** 2 + yis ** 2) / (self.focal_length ** 2) + 1)
        )

        return depth


class ToDepthMap:
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic

    def __call__(self, tensor: torch.Tensor) -> DepthMap:
        depth_map = DepthMap(tensor.float(), self.intrinsic)
        return depth_map


class MaskDepthValues:
    def __init__(self, near: float, far: float):
        self.near = near
        self.far = far

    def __call__(self, depth_map: DepthMap) -> DepthMap:
        mask = (depth_map.depth_map > self.far) | (depth_map.depth_map < self.near)
        depth_map.depth_map[mask] = 0.0
        return depth_map


class ToBinaryMasks:
    def __call__(self, tensor: torch.Tensor, labels: Optional[List] = None) -> Dict[int, torch.Tensor]:
        if labels is None:
            labels = torch.unique(tensor).tolist()
        masks = {}
        for label in labels:
            masks[label] = (tensor == label).bool()

        return masks


class DepthJitter:
    def __init__(self, low=-0.05, high=0.05):
        self.low = low
        self.high = high

    def __call__(self, tensor, *args, **kwargs):
        shift = np.random.uniform(self.low, self.high)

        shifted = tensor - shift

        return shifted


class ToNumpyArray:
    def __call__(self, image):
        return np.array(image)


class ToTensorFromNumpy:
    def __call__(self, image):
        return torch.from_numpy(image)


class Resize:
    def __init__(self, size, mode=Image.NEAREST):
        self.size = size
        self.mode = mode

    def __call__(self, image):
        ow, oh = self.size
        image = image.resize((ow, oh), self.mode)

        return image


class ToImage:
    def __call__(self, array):
        return Image.fromarray(array)


class Lighting:
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, image):
        if self.alphastd == 0:
            return image

        alpha = image.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(image).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        image = image.add(rgb.view(3, 1, 1).expand_as(image))

        return image


class Grayscale:
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation:
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness:
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)

        return img.lerp(gs, alpha)


class Contrast:
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class RandomOrder:
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        if self.transforms is None:
            return image
        order = torch.randperm(len(self.transforms))
        for i in order:
            image = self.transforms[i](image)

        return image


class ColorJitter(RandomOrder):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor:
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, image):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
        image = self.to_tensor(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = self.normalize(image, self.mean, self.std)
        return image

    @staticmethod
    def normalize(tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respectively.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respectively.
        Returns:
            Tensor: Normalized image.
        """

        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor


class RandomHorizontalFlip:
    def __call__(self, image):
        if random.random() < 0.5:
            image = T.functional.hflip(image)

        return image


class RandomRotation:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image):

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        angle = random.uniform(-self.degree, self.degree)
        image = T.functional.rotate(image, angle, False, False, None)

        return image

