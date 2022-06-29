import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(DIM):
    return A.Compose(
        [
            A.RandomResizedCrop(height=DIM, width=DIM),
            A.HorizontalFlip(),
            A.VerticalFlip(),


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
