from dataset import MURADataModule
from model import TestModule

import pytorch_lightning as pl
import torch
from torch import nn, optim, utils

def get_datamodule(name: str, **kwargs) -> pl.LightningDataModule:
    if name == 'MURA':
        return MURADataModule(**kwargs)
    else:
        raise NameError(f'Unknown datamodule: {name}')

def get_model(name: str, **kwargs) -> pl.LightningModule:
    if name == 'test':
        return TestModule(**kwargs)
    else:
        raise NameError(f'Unknown model: {name}')