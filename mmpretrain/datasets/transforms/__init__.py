# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import (CenterCrop, LoadImageFromFile, Normalize,
                             RandomFlip, RandomGrayscale, RandomResize, Resize)

from mmpretrain.registry import TRANSFORMS
from .auto_augment import (AutoAugment, AutoContrast, BaseAugTransform,
                           Brightness, ColorTransform, Contrast, Cutout,
                           Equalize, GaussianBlur, Invert, Posterize,
                           RandAugment, Rotate, Sharpness, Shear, Solarize,
                           SolarizeAdd, Translate)
from .formatting import (Collect, PackInputs, PackMultiTaskInputs, ToNumpy,
                         ToPIL, Transpose)
from .processing import (Albumentations, BEiTMaskGenerator, ColorJitter,
                         EfficientNetCenterCrop, EfficientNetRandomCrop,
                         Lighting, RandomCrop, RandomErasing,
                         RandomResizedCrop, ResizeEdge, SimMIMMaskGenerator, SaveExamples, FilledResize)
from .wrappers import MultiView

for t in (CenterCrop, LoadImageFromFile, Normalize, RandomFlip,
          RandomGrayscale, RandomResize, Resize):
    TRANSFORMS.register_module(module=t)

__all__ = [
    'ToPIL', 'ToNumpy', 'Transpose', 'Collect', 'RandomCrop',
    'RandomResizedCrop', 'Shear', 'Translate', 'Rotate', 'Invert',
    'ColorTransform', 'Solarize', 'Posterize', 'AutoContrast', 'Equalize',
    'Contrast', 'Brightness', 'Sharpness', 'AutoAugment', 'SolarizeAdd',
    'Cutout', 'RandAugment', 'Lighting', 'ColorJitter', 'RandomErasing',
    'PackInputs', 'Albumentations', 'EfficientNetRandomCrop',
    'EfficientNetCenterCrop', 'ResizeEdge', 'BaseAugTransform',
    'PackMultiTaskInputs', 'GaussianBlur', 'BEiTMaskGenerator',
    'SimMIMMaskGenerator', 'CenterCrop', 'LoadImageFromFile', 'Normalize',
    'RandomFlip', 'RandomGrayscale', 'RandomResize', 'Resize', 'MultiView'
]
