import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from skimage import io
from PIL import Image


class HBWDataset(Dataset):
    """
    A dataset object for a computer vision task.

    Args:
        csv_path (str): The path to the CSV file containing the dataset information.
        transform (transforms, optional): Optional data transformation function to be applied to the images.

    Attributes:
        dataset_df (DataFrame): The dataset dataframe read from the CSV file.
        transform (transforms): The transformation function to be applied to the images in the dataset.

    Example:
        >>> dataset = HBWDataset('data.csv', transform=transforms.ToTensor())
    """

    def __init__(self, csv_path: str, transform: transforms = None):
        """
        Initializes the HBWDataset by reading a CSV file, storing the dataset dataframe, and setting the transformation function.

        Args:
            csv_path (str): The path to the CSV file containing the dataset information.
            transform (transforms, optional): Optional data transformation function to be applied to the images.

        Returns:
            None
        """
        super(HBWDataset, self).__init__()
        self.dataset_df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset_df)

    def __getitem__(self, idx):
        """
        Returns the image path at the given index.

        Args:
            idx (int): The index of the image path to retrieve.

        Returns:
            dict: returns a dictionary with X and y values. 'X' is a tuple with (image, gender) and 'y' is a tensor containing SMPLx body vertices.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataset_df.iloc[idx, 0]
        gender = self.dataset_df.iloc[idx, 1]
        smplx_vert_path = self.dataset_df.iloc[idx, 2]

        image = Image.fromarray(io.imread(img_path))
        smplx_verts = torch.from_numpy(np.load(smplx_vert_path))
        if self.transform:
            image = self.transform(image)
        sample = {"X": (image, gender), "y": smplx_verts}
        return sample
