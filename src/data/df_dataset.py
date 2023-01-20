import math

import os.path as osp
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def read_image(path):
    """Reads image from path using ``PIL.Image``.
    Args:
        path (str): path to an image.
    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(
                    path
                )
            )
    return img


def get_labels(data_df):
    names = pd.unique(data_df["name"])

    id_to_labels_dict = {name: label for label, name in enumerate(names)}
    labels = [id_to_labels_dict[name] for name in data_df["name"]]

    return labels


class DFDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

        self.labels = get_labels(self.data)
        self.num_labels = len(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        annot = row["annot"]
        img_path = row["image"]
        name = row["name"]
        x = row["x"]
        y = row["y"]
        w = row["w"]
        h = row["h"]
        label = self.labels[idx]

        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, annot, name
