import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Union
import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.composition import Compose
import torchvision
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch
import math


# These are the mean and std used for image normalization when a network has been trained with ImageNet data.
# If there's enough images in the ImageNet database, then these might reflect 'state of nature', up to certain degree,
# depending on the camera settings.
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# These have been calculated simply from the IMAGE_NET_MEAN and IMAGE_NET_STD. STD is calculated from the average of
# covariances. This are not accurate since the RGB -> grayscale conversion probably weights differently different color channels,
# but we can use these as a first order approximation before getting better values.
IMAGE_NET_MEAN_GREY = 0.449
IMAGE_NET_STD_GREY = 0.226

# Define a set of image augmentations using Albumentations
RAND_8_TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Perspective(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GaussianBlur(p=0.5),
        A.MotionBlur(p=0.5),
        A.PixelDropout(p=0.5, dropout_prob=0.0001),
        A.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
        ToTensorV2(),
    ]
)

RAND_10_TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Perspective(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GaussianBlur(p=0.5),
        A.MotionBlur(p=0.5),
        A.RandomGamma(p=0.5),
        A.ISONoise(p=0.5),
        A.PixelDropout(p=0.5, dropout_prob=0.0001),
        A.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
        ToTensorV2(),
    ]
)

# Normalization transformation that normalizes the input image to ImageNet distribution
IMAGENET_NORMALISATION_TRANSFORM = A.Compose([A.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])


def visualize_transformation(
    image: Union[np.ndarray, str],
    nr_images: int = 15,
    transformation: Union[None, BasicTransform, Compose, torch.nn.Module, torchvision.transforms.Compose] = None,
) -> None:
    """Applies the transformation to the given input image and visualizes the results.

    Parameters
    ----------
    image : Union[np.ndarray, str]
        Either an image (nd.array) or a path to an image to be transformed and visualized
    nr_images : int
        Number of images that will be generated, and visualized, after applying the transformation, by default 15
    transformation : Union[None, BasicTransform, Compose, transforms.Transform], optional
        Transformation to be applied to the image, by default None

    Returns
    -------
    None
    """

    if isinstance(image, str):
        # Read image -> CHW
        image = read_image(image)

    num_rows = math.ceil(nr_images / 3)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i in range(nr_images):
        if transformation:
            if isinstance(transformation, (A.BasicTransform, A.ImageOnlyTransform, A.BaseCompose)):
                image_transformed = transformation(image=image.permute(1, 2, 0).numpy())["image"]
            else:
                image_transformed = transformation(image.to(torch.float32))

        # Expectation here is that the image is in CHW after the transformation
        image_transformed = image_transformed.permute(1, 2, 0).numpy()
        axes[i].imshow(image_transformed)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
