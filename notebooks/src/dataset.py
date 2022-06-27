import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class Cultivar_data(Dataset):

    def __init__(self, image_path, cfg, targets, transform=None):
        self.image_path = image_path
        self.cfg = cfg
        self.transform = transform
        self.targets = targets

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):

        image_path_single = self.image_path[idx]
        if self.cfg['in_channels'] == 1:
            image = cv2.imread(image_path_single, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path_single)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.targets[idx])
        return image, label
