import pandas as pd
from PIL import Image
import pytorch_lightning as pl
import torch
from torch import utils
import torchvision
from torchvision import transforms

import os
from typing import Callable, Dict, Literal, Optional

class MURADataModule(pl.LightningDataModule):
    def __init__(self, root_dir: Optional[str] = None, batch_size: int = 32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()])
        self.valid_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()])
        self.test_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()])

    def setup(self, stage: str):
        self.train_dataset = MURA(
            split='train',
            root_dir=self.root_dir,
            transform=self.train_transforms)

        self.valid_dataset = MURA(
            split='valid',
            root_dir=self.root_dir,
            transform=self.valid_transforms)

        self.test_dataset = MURA(
            split='test',
            root_dir=self.root_dir,
            transform=self.test_transforms)

    def train_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def valid_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

class MURA(utils.data.Dataset):
    def __init__(self, split: Literal['train', 'valid', 'test'], 
                       root_dir: Optional[str] = None, 
                       transform: Optional[Callable] = None):
        
        self.split = split
        if root_dir is None:
            self.root_dir = os.path.join(os.getcwd(), 'data', 'MURA-v1.1')
        else:
            self.root_dir = root_dir

        self.transform = transform

        # TODO: Split train into train and validation (80/20?) and use valid as test set
        self.data = {
            'train': {
                'image_paths': pd.read_csv(os.path.join(self.root_dir, 'train_image_paths.csv'), header=None),
                'image_labels': pd.read_csv(os.path.join(self.root_dir, 'train_labeled_studies.csv'), header=None)
            },
            'valid': {
                'image_paths': pd.read_csv(os.path.join(self.root_dir, 'valid_image_paths.csv'), header=None),
                'image_labels': pd.read_csv(os.path.join(self.root_dir, 'valid_labeled_studies.csv'), header=None)
            },
            # TODO: temp fix for test set --- same as valid
            'test': {
                'image_paths': pd.read_csv(os.path.join(self.root_dir, 'valid_image_paths.csv'), header=None),
                'image_labels': pd.read_csv(os.path.join(self.root_dir, 'valid_labeled_studies.csv'), header=None)
            }
        }

    def __len__(self) -> int:
        return len(self.data[self.split]['image_paths'])

    def __getitem__(self, index: int) -> Dict:
        image_path = self.data[self.split]['image_paths'].loc[index][0]
        study_dir = os.path.dirname(image_path) + '/'
        labels = self.data[self.split]['image_labels']
        label = int(labels.loc[labels[0] == study_dir].values[0][1])
        image_path = os.path.join(os.path.dirname(self.root_dir), image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        sample = {'image': image, 'label': label}
        
        return sample

class ROCO(utils.data.Dataset):

    def __init__(self, split: Literal['train', 'valid', 'test'], 
                       only_radiology: bool = True, 
                       root_dir: Optional[str] = None, 
                       transform: Optional[Callable] = None):
        
        self.split = split
        if root_dir is None:
            self.root_dir = os.path.join(os.getcwd(), 'data', 'ROCO')
        else:
            self.root_dir = root_dir

        # TODO: Handle non-radiology images
        self.only_radiology = only_radiology
        self.transform = transform

        self.data = {
            'train': pd.read_csv(os.path.join(self.root_dir, 'train', 'radiology', 'captions.txt'), header=None, delimiter='\t'),
            'valid': pd.read_csv(os.path.join(self.root_dir, 'validation', 'radiology', 'captions.txt'), header=None, delimiter='\t'),
            'test': pd.read_csv(os.path.join(self.root_dir, 'test', 'radiology', 'captions.txt'), header=None, delimiter='\t'),
        }

    def __len__(self) -> int:
        return len(self.data[self.split])

    def __getitem__(self, index: int) -> Dict:
        image_name, caption = self.data[self.split].loc[index].values
        image_name = image_name + '.jpg'
        caption = caption.strip()
        image_path = os.path.join(self.root_dir, self.split, 'radiology', 'images', image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        sample = {'image': image, 'caption': caption}

        return sample