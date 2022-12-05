import numpy as np
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
import torch
from torch import nn, optim, utils
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
from transformers import AutoTokenizer

import os
import random
import re
from typing import Callable, Dict, List, Literal, Optional

from utils import tokenized_session
from utils import fin_imp_place


class SquarePad(nn.Module):
    # https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/5
    def __init__(self):
        super().__init__()

    def forward(self, image: Image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

class RandomTransformation(nn.Module):

    def __init__(self, family: List[nn.Module]):
        super().__init__()
        self.family = family

    def forward(self, image: Image):
        transform = random.choice(self.family)
        return transform(image)

class DownstreamDataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, root_dir: Optional[str] = None, batch_size: int = 32, num_workers: int = 2, frac: float = 1.0):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.frac = frac
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        self.valid_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def get_dataset(self, name: str, split:str, root_dir: str, transform: Callable) -> utils.data.Dataset:
        if name == 'MURA':
            return MURA(split=split, root_dir=root_dir, transform=transform)
        elif name == 'CHEXPERT':
            return CHEXPERT(split=split, root_dir=root_dir, transform=transform)
        else:
            raise ValueError(f'Unknown dataset {name}')

    def get_dataset_split(self, dataset: utils.data.Dataset, frac: float) -> utils.data.Subset:
        num_elements = int(frac * len(dataset))
        indices = torch.randperm(len(dataset))[:num_elements]
        return utils.data.Subset(dataset, indices)

    def setup(self, stage: str):
        self.train_dataset = self.get_dataset(
            name=self.dataset,
            split='train', 
            root_dir=self.root_dir,
            transform=self.train_transforms)

        self.train_dataset = self.get_dataset_split(
            self.train_dataset, 
            frac=self.frac)

        self.valid_dataset = self.get_dataset(
            name=self.dataset,
            split='valid', 
            root_dir=self.root_dir,
            transform=self.valid_transforms)

        # self.valid_dataset = get_dataset_split(self.valid_dataset, frac=1.0)

        self.test_dataset = self.get_dataset(
            name=self.dataset,
            split='test', 
            root_dir=self.root_dir,
            transform=self.test_transforms)

        # self.valid_dataset = get_dataset_split(self.valid_dataset, frac=1.0)

    def train_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.train_dataset, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers, 
                                     pin_memory=True,
                                     shuffle=True)
                                     # collate_fn=lambda x: tuple(zip(*x)))

    def val_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.valid_dataset, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers, 
                                     pin_memory=True,
                                     shuffle=False)
                                     # collate_fn=lambda x: tuple(zip(*x)))

    def test_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.test_dataset, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers, 
                                     pin_memory=True,
                                     shuffle=False)
                                     # collate_fn=lambda x: tuple(zip(*x)))

class MURA(utils.data.Dataset):
    def __init__(self, split: Literal['train', 'valid', 'test'], 
                       root_dir: Optional[str] = None, 
                       transform: Optional[Callable] = None):
        
        self.split = split

        if root_dir is None:
            self.root_dir = os.path.join(os.getcwd(), 'data', 'MURA-v1.1')
        else:
            self.root_dir = os.path.join(root_dir, 'MURA-v1.1')

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
        if isinstance(index, torch.Tensor):
            index = int(index.item())
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
    def __init__(self, root_dir: Optional[str] = None, batch_size: int = 32, num_workers: int = 2, frac: float = 1.0, text_req : str = 'both'):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frac = frac
        self.text_req = text_req
        self.family = [ 
            transforms.RandomResizedCrop(size=(224, 224),
                                        ratio=(0.6, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomAffine(degrees=20, 
                                    translate=(0.0, 0.1), 
                                    scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=(0.6, 1.4),
                                   contrast=(0.6, 1.4)),
            transforms.GaussianBlur(kernel_size=3,
                                    sigma=(0.1, 3.0),)]

        self.train_transform = {
            'image': transforms.Compose([
                SquarePad(),
                RandomTransformation(family=self.family),
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
        }
        self.valid_transform = {
            'image': transforms.Compose([
                SquarePad(),
                RandomTransformation(family=self.family),
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
        }

    def setup(self, stage: str):
        self.train_dataset = MIMIC_CXR(split='train', text_req=self.text_req, image_transform=self.train_transform['image'])
        #self.valid_dataset = MIMIC_CXR(split='valid', text_req=self.text_req, image_transform=self.valid_transform['image'])

        self.train_dataset = self.get_dataset_split(self.train_dataset, frac=self.frac)
        #self.valid_dataset = self.get_dataset_split(self.valid_dataset, frac=self.frac)

        # self.valid_dataset = get_dataset_split(self.valid_dataset, frac=1.0)
    def get_dataset_split(self, dataset: utils.data.Dataset, frac: float) -> utils.data.Subset:
        num_elements = int(frac * len(dataset))
        indices = torch.randperm(len(dataset))[:num_elements]
        return utils.data.Subset(dataset, indices)

    def train_dataloader(self) -> utils.data.DataLoader:
        return utils.data.DataLoader(self.train_dataset, 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers, 
                                     pin_memory=True,
                                     shuffle=True)

    #def val_dataloader(self) -> utils.data.DataLoader:
    #    return utils.data.DataLoader(self.valid_dataset, 
    #                                 batch_size=self.batch_size, 
    #                                 num_workers=self.num_workers, 
    #                                 pin_memory=True,
    #                                 shuffle=False)

class MIMIC_CXR(utils.data.Dataset):
    def __init__(self, 
        split: Literal['train', 'valid'],
        root_dir_img: str = '/vast/vs2393/mlh_dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/',
        root_dir_txt: str = '/vast/vs2393/mlh_dataset/',
        text_req: str = 'both', # this is helpful in deciding what text we want
        image_transform: Optional[Callable] = None, 
        text_transform: Optional[Callable] = None,
        separate_sections: bool = False):
        self.root_dir_img = root_dir_img
        self.root_dir_txt = root_dir_txt
        self.text_req = text_req
        self.split = split
        self.separate_sections = separate_sections
        self.df_whole = pd.read_csv(os.path.join(self.root_dir_txt, "cxr-record-list.csv"))   ##/home/vs2393/mlh/MLH_Fall22/dataset.py
        row_size = self.df_whole.shape[0]
        self.df = self.df_whole
        with open('missing_reports.txt', 'r') as f:
            lines = f.readlines()
        indices_to_drop = []
        for line in lines:
            index, file_path = line.strip().split()
            indices_to_drop.append(int(index))
        self.df.drop(indices_to_drop, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        #if split == "train":
        #    self.df = self.df_whole.iloc[1:int(row_size*0.7)]
        # else:
        #     self.df = self.df_whole.iloc[int(row_size*0.7):]
        # TODO: rename to actual MIMIC-CXR unzipped directory name
        # if root_dir is None:
        #     self.root_dir = os.path.join(os.getcwd(), 'MIMIC-CXR-TEST')
        # else:
        #     self.root_dir = os.path.join(root_dir, 'MIMIC-CXR-TEST')

        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        # TODO: os walk and split into train and validation (80/20?)
        # self.data = (
        #     ('patient0/0.jpg', 'patient0/report.txt'),
        #     ('patient0/1.jpg', 'patient0/report.txt'),
        #     ('patient1/0.jpg', 'patient1/report.txt'),
        #     ('patient1/1.jpg', 'patient1/report.txt'),
        #     ('patient1/2.jpg', 'patient1/report.txt'))

        # TODO: Text transformations (extracting/splicing out sections)
        #self.text_transform = text_transform
        # TODO: Image transformations (square padding -> resize -> below image transformations)
        #       - cropping, horizontal flipping, affine transformation, color jittering and Gaussian blur
        self.image_transform = image_transform

    def __len__(self) -> int:
        # TODO: Fix with proper splits
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        if isinstance(index, torch.Tensor):
            index = int(index.item())
        image_path_with_dcm = self.df.loc[index]['path']
        image_path_with_jpg = image_path_with_dcm.split(".")[0] + ".jpg"
        image_path = os.path.join(self.root_dir_img, image_path_with_jpg)
        image = Image.open(image_path).convert('RGB')
        if self.image_transform is not None:
            image = self.image_transform(image)
        #print(type(image))
        report_path = "/".join(image_path_with_dcm.split("/")[:-1]) + ".txt"
        report_path = os.path.join(self.root_dir_txt, report_path)

        with open(report_path, 'r') as f:
            lines = f.readlines()

        #Find where FINDINGS and IMPRESSION start
        fin_start, fin_end, imp_start, imp_end = fin_imp_place(lines)

        #Slice the sections
        findings = lines[fin_start: fin_end +1]
        impression = lines[imp_start: imp_end +1]

        #Tokenize
        if self.text_req == "findings":
            if fin_start:
                finding_session = tokenized_session(findings)
                tokenized_finding = self.tokenizer(finding_session, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
                return {'image': image, 'report': tokenized_finding}

        elif self.text_req == "impressions":
            if imp_start:
                impression_session = tokenized_session(impression)
                tokenized_impression = self.tokenizer(impression_session, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
                return {'image': image, 'report': tokenized_impression}

        else:
            if fin_start and imp_start:
                if self.separate_sections:
                    impression = tokenized_session(impression)
                    tokenized_impression = self.tokenizer(impression, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
                    findings = tokenized_session(findings)
                    tokenized_finding = self.tokenizer(findings, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
                    return {'image': image, 'impressions': tokenized_impression, 'findings': tokenized_finding}
                else:
                    imp_fin = findings + impression
                    imp_fin_session = tokenized_session(imp_fin)
                    tokenized_impression = self.tokenizer(imp_fin_session, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
                    return {'image': image, 'report': tokenized_impression}
        
            
class CHEXPERT(utils.data.Dataset):
    def __init__(self, split: Literal['train', 'valid', 'test'],
                       root_dir: Optional[str] = None,
                       transform: Optional[Callable] = None):
        self.split = split
        if root_dir is None:
            self.root_dir = os.path.join(os.getcwd(), 'data', 'CheXpert-v1.0-small')
        else:
            self.root_dir = os.path.join(root_dir, 'CheXpert-v1.0-small')

        self.transform = transform

        # TODO: Add test split (similar to MURA?)
        self.data = {
            'train': pd.read_csv(os.path.join(self.root_dir, 'train.csv')),
            'valid': pd.read_csv(os.path.join(self.root_dir, 'valid.csv'))
        }

    def __len__(self) -> int:
        return len(self.data[self.split])

    def __getitem__(self, index: int) -> Dict:
        if isinstance(index, torch.Tensor):
            index = int(index.item())
        # TODO: Get label
        image_path = self.data[self.split]['Path'].loc[index]
        image_path = os.path.join(os.path.dirname(self.root_dir), image_path)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        sample = {'image': image}

        return sample
