import pandas as pd
import torchvision.io
from torchvision.io import read_image
from torch.utils.data import Dataset, BatchSampler
import torchvision.transforms.functional as TF
import numpy as np
import os
import errno
import torch
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.composition import Compose
import albumentations as A
from ailivesimml.models.model_utils import resize_aspect_ratio


class GroupedBatchSampler(BatchSampler):
    def __init__(self, data_frame, group_key, batch_size):
        self.batch_size = batch_size
        self.groups = {k: np.array(v) for k, v in data_frame.groupby(group_key).groups.items()}

    def __iter__(self):
        group_list = list(self.groups.keys())
        np.random.shuffle(group_list)
        for group in group_list:
            np.random.shuffle(self.groups[group])
            group_indices = self.groups[group]
            for i in range(0, len(group_indices), self.batch_size):
                yield group_indices[i : i + self.batch_size]

    def __len__(self):
        count = 0
        for indices in self.groups.values():
            count += len(indices) // self.batch_size
        return count


class CPSImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        transformation=None,
        grayscale=False,
        degrees=False,
        scaling_factor=0.25,
        aspect_ratio=False,
        target_width=480,
        target_height=288,
    ):
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset.iloc[idx]["image_path"]
        try:
            if self.grayscale:
                image = read_image(path=img_path, mode=torchvision.io.ImageReadMode.GRAY).float() / 255.0
            else:
                image = read_image(img_path).float() / 255.0

            if self.aspect_ratio:
                image, padding = resize_aspect_ratio(image, self.target_width, self.target_height)
            else:
                original_height, original_width = image.shape[1], image.shape[2]
                new_height = int(original_height * self.scaling_factor)
                new_width = int(original_width * self.scaling_factor)
                image = TF.resize(image, size=(new_height, new_width), antialias=True)

            if self.transformation:
                if isinstance(self.transformation, (A.BasicTransform, A.ImageOnlyTransform, A.BaseCompose)):
                    image = self.transformation(image=image.permute(1, 2, 0).numpy())["image"]
                    image = TF.resize(image, size=(self.target_height, self.target_width), antialias=True)
                else:
                    image = self.transformation(image)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

        width = self.dataset.iloc[idx]["width"].astype("float32")
        height = self.dataset.iloc[idx]["height"].astype("float32")
        parameters = self.dataset.iloc[idx][["fov", "k1", "k2", "k3", "p1", "p2", "cx", "cy"]].to_numpy().astype("float32")
        parameters[6] = parameters[6] / width
        if self.aspect_ratio:
            parameters[7] = padding / self.target_height + (self.target_height - 2 * padding) * parameters[7] / (
                height * self.target_height
            )
        else:
            parameters[7] = parameters[7] / height
        if not self.degrees:
            parameters[0] = np.radians(parameters[0])
        # Also return normalized width and 
        assert image.shape == (1, self.target_height, self.target_width), f"Inconsistent image shape: {image.shape}"
        assert parameters.shape == (8,), f"Inconsistent parameters shape: {parameters.shape}"
        
        return image, parameters, torch.tensor([height, width], dtype=torch.float32)
