from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision import transforms


class CTData(Dataset):
    """CT dataset."""

    def __init__(self, root_dir='../data/*.png'):
        """
        Args:
            root_dir: obvious....
        """
        self.root_dir = root_dir
        self.image_paths = sorted([name for name in glob(self.root_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        
        return image.convert('RGB'), 10