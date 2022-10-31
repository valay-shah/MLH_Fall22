import pandas as pd
from PIL import Image
import pytorch_lightning as pl
import torch
from torch import utils
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer

import os
from typing import Callable, Dict, Literal, Optional

class MURADataModule(pl.LightningDataModule):
    def __init__(self, root_dir: Optional[str] = None, batch_size: int = 32, num_workers: int = 2):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        return utils.data.DataLoader(self.train_dataset, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers, 
                                     pin_memory=True,
                                     shuffle=True)

    def val_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.valid_dataset, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers, 
                                     pin_memory=True,
                                     shuffle=False)

    def test_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.test_dataset, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers, 
                                     pin_memory=True,
                                     shuffle=False)

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
        train_val_image_paths = pd.read_csv(os.path.join(self.root_dir, 'train_image_paths.csv'), header=None)
        train_image_paths = train_val_image_paths.sample(frac=0.8, random_state=0).reset_index(drop=True)
        val_image_paths = train_val_image_paths.drop(train_image_paths.index).reset_index(drop=True)
        
        self.data = {
            'train': {
                'image_paths': train_image_paths,
                'image_labels': pd.read_csv(os.path.join(self.root_dir, 'train_labeled_studies.csv'), header=None)
            },
            'valid': {
                'image_paths': val_image_paths,
                'image_labels': pd.read_csv(os.path.join(self.root_dir, 'train_labeled_studies.csv'), header=None)
            },
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

class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: Optional[str] = None, batch_size: int = 32, num_workers: int = 2):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = {
            'image': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
        }
        self.valid_transform = {
            'image': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
        }

    def setup(self, stage: str):
        self.train_dataset = MIMIC_CXR(image_transform=self.train_transform['image'])
        self.valid_dataset = MIMIC_CXR(image_transform=self.valid_transform['image'])

    def train_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.train_dataset, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers, 
                                     pin_memory=True,
                                     shuffle=True)

    def val_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.valid_dataset, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers, 
                                     pin_memory=True,
                                     shuffle=False)

class MIMIC_CXR(utils.data.Dataset):
    def __init__(self, 
        image_transform: Optional[Callable] = None, 
        text_transform: Optional[Callable] = None):
        self.root_dir = os.path.join(os.getcwd(), 'data', 'MIMIC-CXR-TEST')
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        # TODO: os walk and split into train and validation (80/20?)
        self.data = (
            ('patient0/0.jpg', 'patient0/report.txt'),
            ('patient0/1.jpg', 'patient0/report.txt'),
            ('patient1/0.jpg', 'patient1/report.txt'),
            ('patient1/1.jpg', 'patient1/report.txt'),
            ('patient1/2.jpg', 'patient1/report.txt'))

        # TODO: Text transformations (extracting/splicing out sections)
        self.text_transform = text_transform
        # TODO: Image transformations (square padding -> resize -> below image transformations)
        #       - cropping, horizontal flipping, affine transformation, color jittering and Gaussian blur
        self.image_transform = image_transform

    def __len__(self) -> int:
        # TODO: Fix
        return 2

    def __getitem__(self, index: int) -> Dict:
        image_path, report_path = self.data[index]
        image_path = os.path.join(self.root_dir, image_path)
        image = Image.open(image_path).convert('RGB')
        if self.image_transform is not None:
            image = self.image_transform(image)

        report_path = os.path.join(self.root_dir, report_path)
        with open(report_path, 'r') as f:
            lines = f.readlines()
        
        report = ''.join(lines)
        if self.text_transform is not None:
            report = self.text_transform(report)

        tokenized_report = self.tokenizer(report, padding=True, truncation=True, max_length=1024, return_tensors='pt')
        
        return {'image': image, 'report': tokenized_report}    

# TODO: remove?
class CHEXPERT(utils.data.Dataset):
    def __init__(self, split: Literal['train', 'valid'],
                       root_dir: Optional[str] = None,
                       transform: Optional[Callable] = None):
        self.split = split
        if root_dir is None:
            self.root_dir = os.path.join(os.getcwd(), 'data', 'CheXpert-v1.0-small')
        else:
            self.root_dir = root_dir

        self.transform = transform

        self.data = {
            'train': pd.read_csv(os.path.join(self.root_dir, 'train.csv')),
            'valid': pd.read_csv(os.path.join(self.root_dir, 'valid.csv'))
        }

    def __len__(self) -> int:
        return len(self.data[self.split])

    def __getitem__(self, index: int) -> Dict:
        image_path = self.data[self.split]['Path'].loc[index]
        image_path = os.path.join(os.path.dirname(self.root_dir), image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        sample = {'image': image}

        return sample


# TODO: Remove?
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