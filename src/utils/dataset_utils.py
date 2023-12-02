import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .hbw_dataset import HBWDataset
import os
import yaml
from .configs import get_data_configs


def build_transforms(
    is_train: bool = True,
):
    data_config = get_data_configs()
    transform = transforms.Compose(
        [
            transforms.Resize(data_config.img_size),
            transforms.CenterCrop(data_config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(data_config.mean, data_config.std),
        ]
    )
    return transform


def create_hbw_csv(
    gender_path: str, images_dir: str, smplx_vertices_path: str, csv_path: str
) -> int:
    """
    Create a CSV file with image paths, genders, and SMPLx vertices paths.

    Args:
        gender_path (str): The path to the YAML file containing gender information for each subject.
        images_dir (str): The path to the directory containing the images.
        smplx_vertices_path (str): The path to the directory containing the SMPLx vertices files.
        csv_path (str): The path to the output CSV file.

    Returns:
        int: The number of rows in the DataFrame representing the image paths, genders, and SMPLx vertices paths.
    """
    with open(gender_path, "r") as stream:
        try:
            genders_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    hbw_dict = {"ImagePaths": [], "Genders": [], "SMPLxVertPaths": []}

    for root, dirs, imgs in os.walk(images_dir):
        if len(imgs) == 0:
            continue
        for img in imgs:
            img_path = os.path.join(root, img)
            subj_id = root.split("\\")[-2].split("_")[0]
            gender = genders_dict[subj_id]
            smplx_vert_path = os.path.join(smplx_vertices_path, f"{subj_id}.npy")

            hbw_dict["ImagePaths"].append(img_path)
            hbw_dict["Genders"].append(gender)
            hbw_dict["SMPLxVertPaths"].append(smplx_vert_path)
    hbw_df = pd.DataFrame(hbw_dict)
    hbw_df.to_csv(csv_path, index=False)
    return len(hbw_df)


def get_datasets(
    csv_path,
    transform: transforms = None,
    test_size: float = 0.1,
):
    """
    Returns the training and testing datasets from a CSV file.

    Args:
        csv_path (str): The path to the CSV file containing the dataset.
        transform (torchvision.transforms, optional): An optional data transformation to be applied to the dataset.
        test_size (float, optional): An optional float value representing the proportion of the dataset to be used for testing.

    Raises:
        IOError: If the CSV file doesn't exist.

    Returns:
        tuple: A tuple containing the training and testing datasets.
            - trainset (torch.utils.data.Dataset): The training dataset.
            - testset (torch.utils.data.Dataset): The testing dataset.
    """
    if not os.path.isfile(csv_path):
        raise IOError(
            f"CSV file: {csv_path} doesn't exist! Invoke the function - `create_hbw_csv` first."
        )
    hbw_dataset = HBWDataset(csv_path, transform)
    dataset_size = len(hbw_dataset)
    test_size = int(dataset_size * test_size)
    train_size = dataset_size - test_size
    trainset, testset = torch.utils.data.random_split(
        hbw_dataset, [train_size, test_size]
    )
    return trainset, testset


def get_dataloader(
    dataset: Dataset, batch_size: int = 4, type: str = "train", val_size: float = 0.2
):
    """
    Create a data loader for a given dataset.

    Args:
        dataset (Dataset): The dataset object containing the data.
        batch_size (int, optional): The number of samples per batch. Defaults to 4.
        type (str, optional): The type of data loader to create (train or test). Defaults to "train".
        val_size (float, optional): The proportion of the dataset to use for validation (only applicable for train type). Defaults to 0.2.

    Returns:
        DataLoader: A data loader object that can be used to iterate over the dataset in batches during training or testing.
    """
    if type == "train":
        dataset_size = len(dataset)
        val_size = int(dataset_size * val_size)

        indices = list(range(dataset_size))

        np.random.shuffle(indices)
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = data.sampler.SubsetRandomSampler(val_indices)

        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True
        )
        val_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True
        )

        return train_loader, val_loader
    elif type == "test":
        test_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
        return test_loader
    else:
        raise ValueError("Invalid `type` argument. Must be either `train` or `test`.")
