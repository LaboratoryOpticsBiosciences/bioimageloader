"""Transforms for bioimages based on albumentations
"""
from typing import Optional

import albumentations
import cv2
import numpy as np


class ExpandToRGB(albumentations.DualTransform):
    """Make sure image has 3 channels, RGB

    Expand axis of image that has 2 channels to have 3 channels mainly for
    visualization. Albumentations ver.

    """
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 1.0,
        select_mask_channel: Optional[int] = None,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.select_mask_channel = select_mask_channel

    def apply(self, img, **params):
        num_channels = img.shape[-1]
        if num_channels != 2:
            raise ValueError
        stacked = np.concatenate(
            [img, np.zeros_like(img[..., 0])[..., np.newaxis]],
            axis=-1,
        )
        # if dtype is not None:
        #     return stacked.astype(dtype)
        return stacked

    def apply_to_mask(self, img, **params):
        if (ch := self.select_mask_channel) is not None:
            return img[..., ch]
        return img


class HWCToCHW(albumentations.ImageOnlyTransform):
    """Transpose axes
    """
    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # img to have (H, W, C)
        return img.transpose(-1, 0, 1)


class SqueezeGrayImageCHW(albumentations.ImageOnlyTransform):
    """Squeeze grayscale image from (3, H, W) to (1, H, W)

    By default, ``bioimageloader`` returns images in 3 channels regardless of
    their color mode for easy handling. It converts (3, H, W) to (1, H, W).

    Parameters
    ----------
    keep_dim : bool, default: True
    always_apply : bool, default: False
    p : float, default: 1.0
        value between [0.0, 1.0]
    """
    def __init__(
        self,
        keep_dim=True,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.keep_dim = keep_dim

    def apply(self, img, **params):
        if self.keep_dim:
            # img to have (1, H, W)
            return img[0:1]
        # img to have (H, W)
        return img[0]


class SqueezeGrayImageHWC(albumentations.ImageOnlyTransform):
    """Squeeze grayscale image from (H, W, 3) to (H, W)

    By default, ``bioimageloader`` returns images in 3 channels regardless of
    their color mode for easy handling. It converts (H, W, 3) to (H, W).

    Parameters
    ----------
    keep_dim : bool, default: False
    always_apply : bool, default: False
    p : float, default: 1.0
        value between [0.0, 1.0]
    """
    def __init__(
        self,
        keep_dim=False,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.keep_dim = keep_dim

    def apply(self, img, **params):
        if self.keep_dim:
            # img to have (H, W, 1)
            return img[..., 0:1]
        # img to have (H, W)
        return img[..., 0]


class RGBToGray(albumentations.ImageOnlyTransform):
    """ToGray preserve all 3 channels from the input. This transform truncates
    channels dimension.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class ToGrayBySum(albumentations.ImageOnlyTransform):
    """Convert image to gray scale by tacking mean of existing channels

    For 2 channels, multi-modal images, ToGray does not make sense. Normally,
    rgb2gray conversions is a linear sum of RGB values. Just summing with eqaul
    weights would be more correct.


    .. [1] https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray

    """
    def __init__(
            self,
            always_apply: bool = False,
            p: float = 1.0,
            num_channels: Optional[int] = None,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.num_channels = num_channels

    def apply(self, img, **params):
        dtype = img.dtype
        if self.num_channels is not None:
            img = np.sum(img, axis=-1) / self.num_channels
            return img.astype(dtype)
        return np.mean(img, axis=-1).astype(dtype)
