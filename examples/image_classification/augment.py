# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""
import torch
from torchvision import transforms

from timm.data.transforms import RandomResizedCropAndInterpolation

from torchvision import transforms
from torchvision.transforms import Lambda
import random



from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class GrayScale(object):
    """
    Apply GrayScale to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
 
    
    
class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2,activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
    
    
def augmentation_generator(
        input_size=224,
        crop_type="rrc",
        color_jitter=0.3,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        grayscale_prob=1.0,
        solarization_prob=1.0,
        gaussian_blur_prob=1.0,
        random_flip_prob=0.5,
    ):
    primary_tfl = [Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),]
    scale=(0.08, 1.0)
    interpolation='bicubic'
    if crop_type == "rrc":
        primary_tfl += [
            RandomResizedCropAndInterpolation(
                input_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip(p=random_flip_prob)
        ]
    else:
        primary_tfl.append(transforms.Resize(input_size, interpolation=3),)
        if crop_type == "simple":
            primary_tfl.append(transforms.RandomCrop(input_size, padding=4,padding_mode='reflect'))
        primary_tfl.append(transforms.RandomHorizontalFlip(p=random_flip_prob))
    
    secondary_tfl = [transforms.RandomChoice([GrayScale(p=grayscale_prob),
                                              Solarization(p=solarization_prob),
                                              GaussianBlur(p=gaussian_blur_prob)])]
   
    if color_jitter is not None and not color_jitter == 0:
        secondary_tfl.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))
    final_tfl = [
            transforms.ToTensor(),
        ]
    if mean != None:
        final_tfl.append(transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
        )
    return transforms.Compose(primary_tfl+secondary_tfl+final_tfl)

def build_eval_transform(
        input_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        eval_crop_ratio=0.875,
        resize_im=True
    ):
    t = [Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),]
    if resize_im:
        size = int(input_size / eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))

    return transforms.Compose(t)