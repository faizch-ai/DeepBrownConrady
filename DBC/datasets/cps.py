import pandas as pd
import torchvision.io
from torchvision.io import read_image
from torch.utils.data import Dataset, BatchSampler, DataLoader
import torchvision
import numpy as np
import os
import errno
from typing import Tuple, Union
import torch
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.composition import Compose
import albumentations as A
import torchvision.transforms.functional as TF
from DBC.models.model_utils import resize_aspect_ratio


class GroupedBatchSampler(BatchSampler):
    """
    A custom batch sampler for PyTorch DataLoader that groups the data before creating batches.
    This sampler ensures that all samples in each batch come from the same group as defined by a specified key in the DataFrame.

    Args:
    data_frame (pd.DataFrame): DataFrame containing the dataset from which to group and create batches.
    group_key (str): Column name in the DataFrame that is used to group the data.
    batch_size (int): Number of samples to include in each batch.

    Attributes:
    batch_size (int): See Parameters.
    groups (dict): Dictionary storing the indices for each group.
    """

    def __init__(self, data_frame, group_key, batch_size):
        self.batch_size = batch_size
        # Group data by the specified key and store indices
        self.groups = {k: np.array(v) for k, v in data_frame.groupby(group_key).groups.items()}

    def __iter__(self):
        """
        Iterator to yield batches of indices, with each batch containing indices of samples belonging to the same group.

        Yields:
        list[int]: A batch of indices, all of which point to samples that belong to the same group.
        """
        # Shuffle groups
        group_list = list(self.groups.keys())
        np.random.shuffle(group_list)
        for group in group_list:
            np.random.shuffle(self.groups[group])
            group_indices = self.groups[group]
            for i in range(0, len(group_indices), self.batch_size):
                yield group_indices[i : i + self.batch_size]

    def __len__(self):
        """
        Calculates the number of batches available given the group sizes and the batch size.

        Returns:
        int: The number of batches that the sampler can produce.
        """
        count = 0
        for indices in self.groups.values():
            count += len(indices) // self.batch_size
        return count


class CPSImageDataset(Dataset):
    """
    Implements the torch.utils.data.dataset class for the CPS dataset.
    """

    def __init__(
        self,
        dataset: Union[str, pd.DataFrame] = "cps_dataset.csv",
        transformation: Union[None, BasicTransform, Compose, torch.nn.Module, torchvision.transforms.Compose] = None,
        grayscale: bool = False,
        degrees: bool = False,
        scaling_factor: float = 0.25,
        aspect_ratio: bool = False,
        target_width: int = 480,
        target_height: int = 270,
    ) -> None:
        """
        Constructor with parameters for initializing the CPSImageDataset.

        Args:
            dataset (Union[str, pd.DataFrame]):
                The dataset is either a path to a CSV file or a pd.DataFrame that contains the paths to the images and
                the parameters used for generating these images. Defaults to "cps_dataset.csv".
            transformation (Union[None, BasicTransform, Compose, torch.nn.Module, torchvision.transforms.Compose],
                optional): The transformation to be applied to the images. This can be any compatible transformation
                from either PyTorch or albumentations. Defaults to None.
            grayscale (bool, optional):
                If set to True, images will be loaded in grayscale mode. Defaults to False.
            degrees (bool, optional):
                If set to True, the field of view (fov) parameter is treated as being in degrees. If False, it is
                converted to radians. Defaults to False.
            scaling_factor (float, optional):
                A scaling factor used to resize images. This factor is applied to both the width and height of each
                image. Defaults to 0.25.
                aspect_ratio: bool
                    If this flag is True, rescale all the images to 480x270. Defaults to False.
                target_width: int
                    Target width if using aspect_ratio. Defaults to 480.
                target_height: int
                    Target height if using aspect_ratio. Defaults to 270.

        Raises:
            OSError:
                Raised if the CSV file specified in the dataset argument does not exist.
            ValueError:
                Raised if the provided dataset argument is neither a string path to a CSV file nor a pandas DataFrame.
        """
        if isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
        elif isinstance(dataset, str):
            if not os.path.isfile(dataset):
                raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), dataset)
            self.dataset = pd.read_csv(dataset)
        else:
            raise ValueError("dataset needs to be either a 'str' or 'pd.DataFrame' type")
        self.transformation = transformation
        self.grayscale = grayscale
        self.degrees = degrees
        self.scaling_factor = scaling_factor
        self.aspect_ratio = aspect_ratio
        self.target_width = target_width
        self.target_height = target_height

    def __len__(self) -> int:
        """Returns number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]:
        """
        Loads and returns a sample from the dataset at the given index idx. The image is read as a PyTorch tensor.
        If an albumentations transformation is provided, the image is converted to HWC format (numpy.ndarray) and
        transformed. Otherwise, the image tensor is directly returned.

        Args:
            idx : int
                Index to a row in the dataset.

        Returns:
            Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]
                An image and the corresponding parameters. The image type depends on the transformation:
                np.ndarray if an albumentations transformation is applied, otherwise torch.Tensor.
        """
        try:
            img_path = self.dataset.iloc[idx]["image_path"]

            if self.grayscale:
                image = read_image(path=img_path, mode=torchvision.io.ImageReadMode.GRAY).float() / 255.0
            else:
                image = read_image(img_path).float() / 255.0

            # Get original dimensions
            original_height, original_width = image.shape[1], image.shape[2]

            # Compute new dimensions based on the scaling factor
            new_height = int(original_height * self.scaling_factor)
            new_width = int(original_width * self.scaling_factor)
            if self.aspect_ratio:
                image, padding = resize_aspect_ratio(image, self.target_width, self.target_height)
            else:
                # Resize image to new dimensions
                image = TF.resize(image, size=(new_height, new_width), antialias=True)

            if self.transformation:
                if isinstance(self.transformation, (A.BasicTransform, A.ImageOnlyTransform, A.BaseCompose)):
                    image = self.transformation(image=image.permute(1, 2, 0).numpy())["image"]
                else:
                    image = self.transformation(image)

        except Exception as e:
            print(f"Error in __getitem__ on image path {img_path}: {e}")
            raise
        width = self.dataset.iloc[idx]["width"].astype("float32")
        height = self.dataset.iloc[idx]["height"].astype("float32")
        try:
            # Extract parameters following the order in the dataframe
            parameters = self.dataset.iloc[idx][["fov", "k1", "k2", "k3", "p1", "p2", "cx", "cy"]].to_numpy().astype("float32")
            parameters[6] = parameters[6] / width  # converting shift x to a percentage of width
            if self.aspect_ratio:
                # Modify cy so it takes in account the padding due to resize_aspect_ratio
                parameters[7] = padding / self.target_height + (self.target_height - 2 * padding) * parameters[7] / (
                    height * self.target_height
                )
            else:
                parameters[7] = parameters[7] / height  # converting shift y to a percentage of height
            if not self.degrees:
                parameters[0] = np.radians(parameters[0])  # Convert fov from degrees to radians.

        except Exception as e:
            print(f"Error in __getitem__ on parameters for image path: {img_path}: {e}")
            raise
        return image, parameters
