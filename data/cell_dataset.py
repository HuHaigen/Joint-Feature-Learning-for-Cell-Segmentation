import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CellDataset(Dataset):
    r"""A class representing a `Dataset` for Cell Segmentation.

        The `__getitem__` method in this class is expected to return `image`, 
        `label(mask)`, `edge map`, `density map`.
    """

    def __init__(self, root, transform=None, train=True, use_density=0):
        self.use_density = use_density
        self.label_list = []
        self.img_list = []

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        self.transform_label = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.root = os.path.join(root, 'train' if train else 'val')

        self.path_label = os.path.join(self.root, 'label')
        self.path_img = os.path.join(self.root, 'image')
        self.path_corner = os.path.join(self.root, 'corner')
        self.path_density = os.path.join(self.root, 'density')

        self.label_list = os.listdir(self.path_label)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(
            self.path_img, self.label_list[idx].split('.')[0] + '.jpg')).convert('RGB')

        img = self.transform(img)

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        label = Image.open(os.path.join(
            self.path_label, self.label_list[idx])).convert('L')
        mask = self.transform_label(label)

        edge_map = Image.open(os.path.join(self.path_corner,
                                           self.label_list[idx].split('.')[0] + '.jpg')).convert('L')
        edge_map = self.transform_label(edge_map)
        edge_map.squeeze_(0)
        edge_map[edge_map != 0] = 1

        if self.use_density != 0:
            density = Image.open(os.path.join(
                self.path_density, self.label_list[idx].split('.')[0] + '.bmp')).convert('L')
            den = self.transform_label(density)
            den_0 = 1 - den

            den_map = torch.zeros((2, den.shape[1], den.shape[2]))
            den_map[0] = den_0
            den_map[1] = den
            den_map = den_map.float()
        else:
            den_map = 0

        mask.squeeze_(0)

        return img, mask.long(), edge_map, den_map
