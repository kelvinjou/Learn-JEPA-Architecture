"""Datasets.  CIFAR-10 by default; Tiny-ImageNet if you point it at a folder.

I-JEPA's headline claim is that it needs no hand-crafted view augmentations.
That claim is literally true of the released config: the ONLY pretraining
transform is RandomResizedCrop + Normalize.  Horizontal flip, colour jitter and
Gaussian blur are all implemented upstream and all switched off.  We keep that.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


def pretrain_transform(img_size: int, mean, std, crop_scale=(0.3, 1.0)):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=crop_scale, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def eval_transform(img_size: int, mean, std):
    return transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def cifar10(root="./data", img_size=32, train=True, augment=True):
    tf = (pretrain_transform if augment else eval_transform)(img_size, CIFAR_MEAN, CIFAR_STD)
    return datasets.CIFAR10(root=root, train=train, download=True, transform=tf)


def imagefolder(path, img_size=64, augment=True):
    tf = (pretrain_transform if augment else eval_transform)(img_size, IN_MEAN, IN_STD)
    return datasets.ImageFolder(path, transform=tf)


def loader(ds, batch_size, collate_fn=None, shuffle=True, workers=4, drop_last=True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate_fn, drop_last=drop_last, pin_memory=False,
                      persistent_workers=workers > 0)
