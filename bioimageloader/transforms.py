"""Custom transforms for bioimages based on albumentations

"""

from typing import Optional, Tuple, Union

import albumentations
import cv2
import numpy as np


class HWCToCHW(albumentations.ImageOnlyTransform):
    """Transpose axes from (H, W, C) to (C, H, W)

    By default, ``bioimageloader`` returns images in shape of (H, W, C=3)
    regardless of its color mode for easy handling. Some models expect (C, H, W)
    shape of images as input. It converts (C, H, W) to (C, H, W).

    See Also
    --------
    albumentations.ImageOnlyTransform : super class

    Examples
    --------
    >>> import albumentations as A
    >>> from bioimageloader import Config, datasets_from_config

    >>> cfg = Config('config.yml')
    >>> transforms = A.Compose([
            HWCToCHW(),
        ])
    >>> dataset = datasets_from_config(cfg, transforms=transforms)
    >>> data = dataset[0]
    >>> print(data['image'].shape)
    (3, H, W)

    """
    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # img to have (H, W, C)
        return img.transpose(-1, 0, 1)


class SqueezeGrayImageCHW(albumentations.ImageOnlyTransform):
    """Squeeze grayscale image from (3, H, W) to (1, H, W)|(H, W)

    By default, ``bioimageloader`` returns images in 3 channels regardless of
    their color mode for easy handling. If a model requires (C=1, H, W) shape of
    input, first use ``HWCToCHW`` and use this transform to convert from (3, H,
    W) to (1, H, W).

    Parameters
    ----------
    keep_dim : bool, default: True
        Keep channel axis to 1
    always_apply : bool, default: False
    p : float, default: 1.0
        Value between [0.0, 1.0]

    See Also
    --------
    albumentations.ImageOnlyTransform : super class
    bioimageloader.transforms.HWCToCHW
    bioimageloader.transforms.SqueezeGrayImageHWC

    Examples
    --------
    >>> import albumentations as A
    >>> from bioimageloader import Config, datasets_from_config

    >>> cfg = Config('config.yml')
    >>> transforms = A.Compose([
            HWCToCHW(),
            SqueezeGrayImageCHW(),
        ])
    >>> dataset = datasets_from_config(cfg, transforms=transforms)
    >>> data = dataset[0]
    >>> print(data['image'].shape)
    (1, H, W)

    You can set ``keep_dim`` False to entirely drop channel axis, as some models
    require. But use ``SqueezeGrayImageHWC`` instead for that.

    >>> transforms = A.Compose([
            HWCToCHW(),
            SqueezeGrayImageCHW(keep_dim=False),  # drop channel axis
        ])
    >>> dataset = datasets_from_config(cfg, transforms=transforms)
    >>> data = dataset[0]
    >>> print(data['image'].shape)
    (H, W)

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

    def get_transform_init_args_names(self):
        return (
            'keep_dim',
        )


class SqueezeGrayImageHWC(albumentations.ImageOnlyTransform):
    """Squeeze grayscale image from (H, W, 3) to (H, W)|(H, W, 1)

    By default, ``bioimageloader`` returns images in 3 channels regardless of
    their color mode for easy handling. It converts (H, W, 3) to (H, W).

    If a model requires (H, W) shape of input, use this transform to convert
    from (H, W, 3) to (H, W).

    Parameters
    ----------
    keep_dim : bool, default: False
    always_apply : bool, default: False
    p : float, default: 1.0
        Value between [0.0, 1.0]

    See Also
    --------
    albumentations.ImageOnlyTransform : super class
    bioimageloader.transforms.SqueezeGrayImageHWC

    Examples
    --------
    >>> import albumentations as A
    >>> from bioimageloader import Config, datasets_from_config

    >>> cfg = Config('config.yml')
    >>> transforms = A.Compose([
            SqueezeGrayImageHWC(),
        ])
    >>> dataset = datasets_from_config(cfg, transforms=transforms)
    >>> data = dataset[0]
    >>> print(data['image'].shape)
    (H, W)

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

    def get_transform_init_args_names(self):
        return (
            'keep_dim',
        )


class ExpandToRGB(albumentations.DualTransform):
    """Make sure image has 3 channels, RGB

    Expand axis of image that has 2 channels to have 3 channels mainly for
    visualization. Albumentations ver.

    Warnings
    --------
    This will be deprecated. Grayscale conversion is done in eash Dataset

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

    def get_transform_init_args_names(self):
        return (
            'select_mask_channel',
        )


class RGBToGray(albumentations.ImageOnlyTransform):
    """ToGray preserve all 3 channels from the input. This transform truncates
    channels dimension.

    Warnings
    --------
    This will be deprecated. Grayscale conversion is done in eash Dataset

    Notes
    -----
    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def get_transform_init_args_names(self):
        return ()


class ToGrayBySum(albumentations.ImageOnlyTransform):
    """Convert image to gray scale by tacking mean of existing channels

    For 2 channels, multi-modal images, ToGray does not make sense. Normally,
    rgb2gray conversions is a linear sum of RGB values (opencv [1]_, pillow
    [2]_). Just summing with eqaul weights would be more correct for bioimages.

    Warnings
    --------
    This will be deprecated. Grayscale conversion is done through ``grayscale``
    and ``grayscale_mode`` arguments in eash Dataset

    References
    ----------
    .. [1] https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
    .. [2] https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert

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


class ChannelReorder(albumentations.ImageOnlyTransform):
    """Reorder channel

    Expect images with 3 channels. Reorder and make it continuous in 'C' order.

    Parameters
    ----------
    order : tuple of three integers
        Reorder by indexing
    always_apply : bool, default: False
    p : float, default: 1.0
        Value between [0.0, 1.0]

    See Also
    --------
    albumentations.ImageOnlyTransform : super class
    albumentations.augmentations.transforms.ChannelShuffle : random shuffling

    Examples
    --------
    >>> import numpy as np
    >>> from bioimageloader.transforms import ChannelReorder

    >>> arr = np.arange(12).reshape((2, 2, 3))
    >>> print(arr)
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]

    >>> reorder = ChannelReorder((2, 1, 0))
    >>> arr_reordered = reorder.apply(arr)
    >>> print(arr_reordered)
    [[[ 2  1  0]
      [ 5  4  3]]
     [[ 8  7  6]
      [11 10  9]]]

    """
    def __init__(
        self,
        order: Tuple[int, int, int],
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.order = order

    def apply(self, img, **params):
        return np.ascontiguousarray(img[..., self.order])

    def get_transform_init_args_names(self):
        return (
            'order',
        )


class NormalizePercentile(albumentations.ImageOnlyTransform):
    """Normalize using percentile

    Compute q-th percentile min- and max-values from given image array and
    normalize.

    Use ``numpy.percentile()`` [1]_

    Expect images with 3 channels. Reorder and make it continuous in 'C' order.

    Parameters
    ----------
    qmin : float, default: 0.0
        Lower bound quantile in range of [0, 100)
    qmax : float, default: 99.8
        Upper bound quantile in range of (0, 100]
    per_channel : bool, default: False
        Whether to calculate percentile per channel or not
    clip : bool, default: False
        Whether to clip in [0, 1] or not. Read more in Returns section.
    order : tuple of three integers
        Reorder by indexing
    always_apply : bool, default: False
    p : float, default: 1.0
        Value between [0.0, 1.0]

    Returns
    -------
    img_norm
        Normalized image in float32 in range of [0.0, 1.0] if ``clip`` set to
        True, else its value overflows lower beyond 0.0 and higher beyond 1.0.

    See Also
    --------
    albumentations.ImageOnlyTransform : super class
    albumentations.augmentations.transforms.ChannelShuffle : random shuffling

    References
    ----------
    .. [1] https://numpy.org/doc/stable/reference/generated/numpy.percentile.html

    Examples
    --------
    >>> import numpy as np
    >>> from bioimageloader.transforms import ChannelReorder

    >>> arr = np.arange(12).reshape((2, 2, 3))
    >>> print(arr)
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]

    >>> reorder = ChannelReorder((2, 1, 0))
    >>> arr_reordered = reorder.apply(arr)
    >>> print(arr_reordered)
    [[[ 2  1  0]
      [ 5  4  3]]
     [[ 8  7  6]
      [11 10  9]]]

    """
    def __init__(
        self,
        qmin: float = 0.0,
        qmax: float = 99.8,
        per_channel: bool = False,
        clip: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.qmin = qmin
        self.qmax = qmax
        self.per_channel = per_channel
        self._per_channel = (0, 1) if per_channel else None
        self.clip = clip

    def apply(self, img, **params):
        if self.qmin == 0:
            vmin = 0.0
            vmax = np.percentile(img, self.qmax,
                                 axis=self._per_channel, keepdims=True)
        else:
            v = np.percentile(img, (self.qmin, self.qmax),
                              axis=self._per_channel, keepdims=True)
            vmin = v[0]
            vmax = v[1]
        img = (img - vmin) / (vmax - vmin)
        if self.clip:
            img = np.clip(img, 0, 1)
        return img.astype(np.float32)

    def get_transform_init_args_names(self):
        return (
            'qmin',
            'qmax',
            'per_channel',
            'clip',
        )
