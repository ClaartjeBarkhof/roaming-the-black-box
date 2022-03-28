# Adapted from: https://github.com/AntixK/PyTorch-VAE/blob/master/dataset.py

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image


class SyntheticDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 transform: Callable,
                 **kwargs):

        assert os.path.exists(data_dir), f"{data_dir} provided path does not exist."

        self.data_dir = Path(data_dir)
        self.transforms = transform

        if "celeb" in str(self.data_dir):
            imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        else:
            imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.png'])

        self.imgs = imgs[:int(len(imgs) * 0.9)] if split == "train" else imgs[int(len(imgs) * 0.9):]

        print(split, " len(imgs) = ", len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class ImageDataset():
    def __init__(
            self,
            data_dir,
            batch_size=8,
            image_dim=64,
            num_workers=0,
            pin_memory=False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.dataset_name = self.data_dir.split("/")[-1]
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Set in set-up
        self.transforms = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_name in ["synthetic_simple_bw" "synthetic_simple_bw_filled"]:
            print("With normalisation!")
            self.transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.Resize(self.image_dim),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(0.476, 0.494)])
        elif self.dataset_name == "img_align_celeba":
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.CenterCrop(148),
                                                  transforms.Resize(self.image_dim),
                                                  transforms.ToTensor(),])
        elif self.dataset_name == "synthetic_random":
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.Resize(self.patch_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5330, 0.5264, 0.5296),
                                                                        (0.3773, 0.3776, 0.3732))])
        else:
            print("No normalisation! Using standard transforms only: RandomHorizontalFlip, Resize & ToTensor.")
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.Resize(self.image_dim),
                                                  transforms.ToTensor(),
                                                  ])  # transforms.Normalize(mean_mean, mean_std)

        self.train_dataset = SyntheticDataset(
            self.data_dir,
            split='train',
            transform=self.transforms,
        )

        self.val_dataset = SyntheticDataset(
            self.data_dir,
            split='val',
            transform=self.transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

def infer_mean_std_dataset(image_dir, n_samples=1000):
    means, stds = [], []
    for i, f in enumerate(os.listdir(image_dir)):
        p = Path(image_dir) / f

        img = np.array(Image.open(p))

        if i == 0:
            print("image shape", img.shape)

        img = img / 255.

        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))

        if i + 1 == n_samples:
            break

    means = np.stack(means)
    stds = np.stack(stds)

    mean_mean = np.mean(means, axis=0)
    mean_std = np.mean(stds, axis=0)

    print("mean_mean shape", mean_mean.shape)
    print("mean_std shape", mean_std.shape)

    return mean_mean, mean_std