import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import numpy as np


def get_train_transforms(DIM):
    return A.Compose(
        [A.CLAHE(),
         A.RandomResizedCrop(height=DIM, width=DIM),
         A.HorizontalFlip(),
         A.VerticalFlip(),

         A.CLAHE(),

         A.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225],
         ),

         ToTensorV2(),
         ]
    )


def get_train_transforms_rgn(DIM):
    return A.Compose(
        [
            A.RandomResizedCrop(height=DIM, width=DIM),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.CLAHE(),

            A.Normalize(
                mean=[0.16479976, 0.24128053, 0.55933255],
                std=[0.17945148, 0.33298728, 0.24033068],
            ),

            ToTensorV2(),
        ]
    )


def get_valid_transforms(DIM):
    return A.Compose(
        [
            A.Resize(height=DIM, width=DIM),
            A.CLAHE(),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),

            ToTensorV2(),
        ]
    )


def get_valid_transforms_rgn(DIM):
    return A.Compose(
        [
            A.Resize(height=DIM, width=DIM),

            A.Normalize(
                mean=[0.16479976, 0.24128053, 0.55933255],
                std=[0.17945148, 0.33298728, 0.24033068],
            ),

            ToTensorV2(),
        ]
    )


def mixup_data(x, y, alpha=1.0, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
