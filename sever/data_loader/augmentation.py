import abc
import random
from copy import deepcopy

import numpy as np
import cv2

from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    RandomBrightnessContrast,
    RandomSizedCrop,
    CoarseDropout,
    RandomCrop,
    RandomRotate90,
    CropNonEmptyMaskIfExists,
    OneOf,
    ImageOnlyTransform,
    GaussianBlur,
    Sharpen,
)


class AugmentationBase(abc.ABC):
    MEAN = (0.3439,)
    STD = (0.0383,)

    H = 256
    W = 1600

    def __init__(self):
        self.transform = self.notimplemented

    def build_transforms(self, train):
        if train:
            self.transform = self.build_train()
        else:
            self.transform = self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    def build_test(self):
        return Compose(
            [
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )

    def notimplemented(self, *args, **kwargs):
        raise Exception("You must call `build_transforms()` before using me!")

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def copy(self):
        return deepcopy(self)


class LightTransforms(AugmentationBase):
    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose(
            [
                HorizontalFlip(p=0.5),
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )


class MediumTransforms(AugmentationBase):
    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Normalize(mean=self.MEAN, std=self.STD),
                RandomBrightnessContrast(p=0.1),
                ToTensorV2(),
            ]
        )


class HeavyTransforms(AugmentationBase):
    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Normalize(mean=self.MEAN, std=self.STD),
                RandomBrightnessContrast(p=0.2),
                RandomSizedCrop((240, 256), self.H, self.W, w2h_ratio=1600 / 256),
                CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(32, 32),
                    hole_width_range=(32, 32),
                    fill=0,
                ),
                ToTensorV2(),
            ]
        )


class RandomCropTransforms(AugmentationBase):
    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose(
            [
                RandomCrop(self.H, self.H),
                HorizontalFlip(p=0.5),
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )


class RandomCropMediumTransforms(AugmentationBase):
    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose(
            [
                RandomCrop(self.H, self.H),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )


class RandomCrop256x400Transforms(AugmentationBase):
    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose(
            [
                RandomCrop(self.H, 416),
                HorizontalFlip(p=0.5),
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )


class HeavyCropTransforms(AugmentationBase):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def build_train(self):
        return Compose(
            [
                OneOf(
                    [
                        CropNonEmptyMaskIfExists(self.height, self.width),
                        RandomCrop(self.height, self.width),
                    ],
                    p=1,
                ),
                OneOf(
                    [
                        CLAHE(p=0.5),  # modified source to get this to work
                        GaussianBlur(3, p=0.3),
                        Sharpen(alpha=(0.2, 0.3), p=0.3),
                    ],
                    p=1,
                ),
                HorizontalFlip(p=0.5),
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )


class HeavyCropClasTransforms(AugmentationBase):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def build_train(self):
        return Compose(
            [
                OneOf(
                    [
                        CropNonEmptyMaskIfExists(self.height, self.width),
                        RandomCrop(self.height, self.width),
                    ],
                    p=1,
                ),
                OneOf(
                    [
                        CLAHE(p=0.5),  # modified source to get this to work
                        GaussianBlur(3, p=0.3),
                        Sharpen(alpha=(0.2, 0.3), p=0.3),
                    ],
                    p=1,
                ),
                HorizontalFlip(p=0.5),
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )

    def build_test(self):
        return Compose(
            [
                RandomCrop(
                    self.height, self.width
                ),  # not fully conv, so need to limit img size
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )


class MaskCropTransforms(AugmentationBase):
    def __init__(self):
        super().__init__()

    def build_train(self):
        return Compose(
            [
                CropNonEmptyMaskIfExists(self.H, self.H),
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )


# -- custom --


class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self, clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5
    ):
        super(CLAHE, self).__init__(always_apply, p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = tuple(tile_grid_size)

    def apply(self, img, clip_limit=2, **params):
        return clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:
        img = clahe.apply(img[:, :, 0])
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # img[:, :, 0] = clahe.apply(img[:, :, 0])
        # img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img[:, :, np.newaxis]


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple([bias + x for x in param])

    return tuple(param)


class StrongMixUpTransforms(AugmentationBase):
    """
    Strong augmentation for class-imbalanced steel defect detection.
    Includes: spatial transforms, intensity augmentation, MixUp-ready.
    Designed to help rare classes (Class 1: crazing, 247 patches)
    by generating diverse training samples through heavy augmentation.
    """

    def __init__(self, height=256, width=384):
        super().__init__()
        self.height = height
        self.width = width

    def build_train(self):
        return Compose(
            [
                # Spatial augmentations - crop around defects
                OneOf(
                    [
                        CropNonEmptyMaskIfExists(self.height, self.width),
                        RandomCrop(self.height, self.width),
                    ],
                    p=1,
                ),
                RandomRotate90(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.3),
                # Color/intensity augmentations for steel surface
                OneOf(
                    [
                        CLAHE(clip_limit=4.0, p=0.5),
                        GaussianBlur(blur_limit=(3, 5), p=0.3),
                        Sharpen(alpha=(0.2, 0.4), p=0.3),
                    ],
                    p=0.6,
                ),
                RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.3,
                ),
                # Texture augmentation - Gaussian noise
                GaussNoise(
                    var_limit=(5.0, 25.0),
                    p=0.25,
                ),
                # Cutout-style augmentation for robustness
                CoarseDropout(
                    num_holes_range=(2, 4),
                    hole_height_range=(16, 48),
                    hole_width_range=(16, 48),
                    fill=0.0,
                    p=0.4,
                ),
                Normalize(mean=self.MEAN, std=self.STD),
                ToTensorV2(),
            ]
        )


class GaussNoise(ImageOnlyTransform):
    """Add Gaussian noise to image — adapted for steel texture."""

    def __init__(self, var_limit=(5.0, 30.0), mean=0.0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.var_limit = var_limit
        self.mean = mean

    def apply(self, img, var=10.0, **params):
        sigma = var**0.5
        noise = np.random.randn(*img.shape) * sigma
        return np.clip(img + noise, 0, 255).astype(img.dtype)

    def get_params(self):
        return {"var": random.uniform(self.var_limit[0], self.var_limit[1])}

    def get_transform_init_args_names(self):
        return ("var_limit", "mean")
