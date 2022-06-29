import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2


class Cultivar_data(Dataset):

    def __init__(self, image_path, cfg, targets, transform=None, transform_rgn=None):
        self.image_path = image_path
        self.cfg = cfg
        self.transform = transform
        self.targets = targets
        self.transform_rgn = transform_rgn

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path_single = self.image_path[idx]
        image_path_single_rgn = image_path_single.replace('.jpg', '_rgn.jpg')

        image = cv2.imread(image_path_single)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        image_rgn = cv2.imread(image_path_single_rgn)
        image_rgn = cv2.cvtColor(image_rgn, cv2.COLOR_BGR2RGB)
        image_rgn = np.array(image_rgn)

        if self.transform is not None:
            image = self.transform(image=image)['image']
            image_rgn = self.transform_rgn(image=image_rgn)['image']
        label = torch.tensor(self.targets[idx])
        return image, image_rgn, label


class Cultivar_data_inference(Dataset):

    def __init__(self, image_path, cfg, transform=None, transform_rgn=None):
        self.image_path = image_path
        self.cfg = cfg
        self.transform = transform
        self.transform_rgn = transform_rgn

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path_single = self.image_path[idx]
        image_path_single_rgn = image_path_single.replace('.jpg', '_rgn.jpg')

        image = cv2.imread(image_path_single)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        image_rgn = cv2.imread(image_path_single_rgn)
        image_rgn = cv2.cvtColor(image_rgn, cv2.COLOR_BGR2RGB)
        image_rgn = np.array(image_rgn)

        if self.transform is not None:
            image = self.transform(image=image)['image']
            image_rgn = self.transform_rgn(image=image_rgn)['image']
        return image, image_rgn
